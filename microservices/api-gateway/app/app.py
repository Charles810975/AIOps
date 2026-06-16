import os
import time
import logging
import requests
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BACKEND_HOST = os.getenv('BACKEND_HOST', 'frontend')
BACKEND_PORT = int(os.getenv('BACKEND_PORT', '8080'))
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '60'))

# Prometheus metrics
REQUEST_COUNT = Counter(
    'api_gateway_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'api_gateway_request_latency_seconds',
    'Request latency',
    ['method', 'endpoint']
)
ACTIVE_CONNECTIONS = Gauge(
    'api_gateway_active_connections',
    'Number of active connections'
)
RATE_LIMIT_HITS = Counter(
    'api_gateway_rate_limit_hits_total',
    'Total rate limit hits'
)
BACKEND_HEALTH = Gauge(
    'api_gateway_backend_health',
    'Backend health status (1=healthy, 0=unhealthy)'
)

# Rate limiting storage
request_counts = defaultdict(list)
rate_lock = Lock()

# Upstream service endpoints
UPSTREAM_SERVICES = {
    'frontend': f'http://{BACKEND_HOST}:{BACKEND_PORT}',
    'productcatalog': 'http://productcatalogservice:3550',
    'cartservice': 'http://cartservice:7070',
    'recommendationservice': 'http://recommendationservice:8080',
}


def is_rate_limited(client_ip):
    now = datetime.now()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)

    with rate_lock:
        request_counts[client_ip] = [
            ts for ts in request_counts[client_ip] if ts > window_start
        ]
        request_counts[client_ip].append(now)

        if len(request_counts[client_ip]) > RATE_LIMIT_REQUESTS:
            RATE_LIMIT_HITS.inc()
            return True
        return False


def get_client_ip():
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'


@app.route('/')
def index():
    return jsonify({
        'service': 'API Gateway',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'endpoints': {
            'health': '/health',
            'metrics': '/metrics',
            'route': '/route/<service>/<path:subpath>',
            'status': '/status',
            'rate_limit_info': '/rate-limit-info'
        }
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'api-gateway',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'upstream_services': list(UPSTREAM_SERVICES.keys())
    })


@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/status')
def status():
    client_ip = get_client_ip()
    with rate_lock:
        req_count = len(request_counts[client_ip])
        active_ips = sum(1 for v in request_counts.values() if v)

    return jsonify({
        'client_ip': client_ip,
        'requests_in_window': req_count,
        'your_request_count': req_count,
        'active_ips': active_ips,
        'rate_limit': RATE_LIMIT_REQUESTS,
        'window_seconds': RATE_LIMIT_WINDOW,
        'backend_host': BACKEND_HOST,
        'backend_port': BACKEND_PORT,
        'upstream_services': list(UPSTREAM_SERVICES.keys())
    })


@app.route('/rate-limit-info')
def rate_limit_info():
    return jsonify({
        'limit': RATE_LIMIT_REQUESTS,
        'window_seconds': RATE_LIMIT_WINDOW,
        'strategy': 'sliding_window'
    })


@app.route('/route/<service>/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def route_request(service, subpath):
    start_time = time.time()

    if service not in UPSTREAM_SERVICES:
        REQUEST_COUNT.labels(method=request.method, endpoint=f'route/{service}', status=404).inc()
        return jsonify({'error': f'Unknown service: {service}'}), 404

    if is_rate_limited(get_client_ip()):
        REQUEST_COUNT.labels(method=request.method, endpoint=f'route/{service}', status=429).inc()
        return jsonify({
            'error': 'Rate limit exceeded',
            'limit': RATE_LIMIT_REQUESTS,
            'window_seconds': RATE_LIMIT_WINDOW
        }), 429

    ACTIVE_CONNECTIONS.inc()
    BACKEND_HEALTH.set(1)

    try:
        upstream_url = f"{UPSTREAM_SERVICES[service]}/{subpath}"
        headers = {k: v for k, v in request.headers if k.lower() not in ('host',)}
        headers['X-Forwarded-By'] = 'api-gateway'
        headers['X-Request-ID'] = request.headers.get('X-Request-ID', f'gw-{int(time.time()*1000)}')

        if request.method == 'GET':
            resp = requests.get(upstream_url, headers=headers, params=request.args, timeout=5)
        elif request.method == 'POST':
            resp = requests.post(upstream_url, headers=headers, data=request.data, timeout=5)
        elif request.method == 'PUT':
            resp = requests.put(upstream_url, headers=headers, data=request.data, timeout=5)
        elif request.method == 'DELETE':
            resp = requests.delete(upstream_url, headers=headers, timeout=5)
        else:
            resp = requests.request(request.method, upstream_url, headers=headers, timeout=5)

        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method=request.method, endpoint=f'route/{service}').observe(latency)
        REQUEST_COUNT.labels(method=request.method, endpoint=f'route/{service}', status=resp.status_code).inc()

        return Response(resp.content, status=resp.status_code,
                        headers=dict(resp.headers), mimetype=resp.headers.get('Content-Type', 'application/json'))

    except requests.exceptions.Timeout:
        BACKEND_HEALTH.set(0)
        REQUEST_COUNT.labels(method=request.method, endpoint=f'route/{service}', status=504).inc()
        return jsonify({'error': 'Gateway timeout - upstream service did not respond'}), 504
    except requests.exceptions.RequestException as e:
        BACKEND_HEALTH.set(0)
        logger.error(f"Upstream error: {e}")
        REQUEST_COUNT.labels(method=request.method, endpoint=f'route/{service}', status=502).inc()
        return jsonify({'error': 'Bad gateway - upstream service error'}), 502
    finally:
        ACTIVE_CONNECTIONS.dec()


GATEWAY_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>API Gateway Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }
  header { background: linear-gradient(135deg, #065f46 0%, #1e3a8a 100%); padding: 24px 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
  h1 { margin: 0 0 6px 0; font-size: 26px; }
  .sub { color: #cbd5e1; font-size: 14px; }
  main { padding: 24px 32px; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; border-radius: 10px; padding: 20px; border-left: 4px solid #10b981; box-shadow: 0 1px 4px rgba(0,0,0,0.3); }
  .card.warn { border-left-color: #f59e0b; }
  .card.info { border-left-color: #3b82f6; }
  .card .label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 8px; }
  .card .value { font-size: 32px; font-weight: 700; }
  .section { background: #1e293b; border-radius: 10px; padding: 20px; margin-bottom: 24px; }
  .section h2 { margin-top: 0; font-size: 18px; border-bottom: 1px solid #334155; padding-bottom: 10px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #334155; font-size: 14px; }
  th { color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
  .svc { display: flex; align-items: center; gap: 10px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .dot.ok { background: #10b981; box-shadow: 0 0 6px #10b981; }
  .dot.bad { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
  code { background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 12px; color: #fca5a5; }
  .actions a { display: inline-block; margin-right: 12px; color: #93c5fd; text-decoration: none; font-size: 13px; }
  .actions a:hover { text-decoration: underline; }
  .refresh { float: right; font-size: 12px; color: #94a3b8; }
</style>
</head>
<body>
<header>
  <h1>API Gateway Dashboard</h1>
  <div class="sub">Online Boutique AIOps Extension · Routing & Rate Limiting · <span id="ts"></span></div>
  <div class="actions" style="margin-top:10px;">
    <a href="/health">/health</a>
    <a href="/status">/status</a>
    <a href="/metrics">/metrics</a>
    <a href="/rate-limit-info">/rate-limit-info</a>
    <a href="/route/frontend/">route -> frontend</a>
  </div>
</header>
<main>
  <div class="grid">
    <div class="card"><div class="label">Active IPs Tracked</div><div class="value" id="ips">-</div></div>
    <div class="card warn"><div class="label">Your Requests (60s)</div><div class="value" id="myreq">-</div></div>
    <div class="card info"><div class="label">Rate Limit</div><div class="value" id="limit">-</div></div>
  </div>
  <div class="section">
    <h2>Upstream Services</h2>
    <table>
      <thead><tr><th>Name</th><th>Backend URL</th><th>Health</th></tr></thead>
      <tbody id="upstream"></tbody>
    </table>
  </div>
  <div class="section">
    <h2>How to Use</h2>
    <p style="color:#cbd5e1;font-size:14px;line-height:1.8">
      The gateway routes requests to upstream microservices. Try:
    </p>
    <ul style="color:#cbd5e1;font-size:13px;line-height:2">
      <li><code>GET /route/frontend/</code> &mdash; forward to <code>frontend:8080</code></li>
      <li><code>GET /route/productcatalog/products</code> &mdash; list products</li>
      <li><code>GET /route/cartservice/cart</code> &mdash; current cart</li>
      <li><code>GET /route/recommendationservice/recommendations</code> &mdash; recommendations</li>
    </ul>
  </div>
</main>
<script>
async function load() {
  try {
    const [status, rli, health] = await Promise.all([
      fetch('/status').then(r=>r.json()),
      fetch('/rate-limit-info').then(r=>r.json()),
      fetch('/health').then(r=>r.json())
    ]);
    document.getElementById('ips').textContent = status.active_ips || 0;
    document.getElementById('myreq').textContent = status.your_request_count || 0;
    document.getElementById('limit').textContent = (rli.limit || '?') + ' / ' + (rli.window_seconds || '?') + 's';
    const upstreams = health.upstream_services || [];
    document.getElementById('upstream').innerHTML = upstreams.map(s=>{
      const isFrontend = s === 'frontend';
      const url = isFrontend ? 'http://' + s + ':8080' : 'http://' + s + ':' + (s==='productcatalog' ? 3550 : s==='cartservice' ? 7070 : 8080);
      return `<tr>
        <td><div class="svc"><span class="dot ok"></span>${s}</div></td>
        <td><code>${url}</code></td>
        <td>configured</td>
      </tr>`;
    }).join('');
    document.getElementById('ts').textContent = 'updated ' + new Date().toISOString().substring(11,19) + ' UTC';
  } catch (e) { console.error(e); }
}
load();
setInterval(load, 5000);
</script>
</body>
</html>"""


@app.route('/dashboard')
@app.route('/dashboard/')
def gateway_dashboard():
    return Response(GATEWAY_DASHBOARD_HTML, mimetype='text/html')


if __name__ == '__main__':
    logger.info(f"API Gateway starting on port {os.getenv('PORT', 8080)}")
    logger.info(f"Routing to backend: {BACKEND_HOST}:{BACKEND_PORT}")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
