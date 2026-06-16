import os
import time
import json
import logging
import threading
from datetime import datetime
from collections import deque
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MAX_ALERTS = int(os.getenv('MAX_ALERTS', '1000'))
ALERT_RETENTION_SECONDS = int(os.getenv('ALERT_RETENTION_SECONDS', '3600'))

# Prometheus metrics
ALERTS_SENT = Counter(
    'notification_alerts_sent_total',
    'Total alerts sent',
    ['severity', 'channel']
)
ALERTS_TRIGGERED = Counter(
    'notification_alerts_triggered_total',
    'Total alerts triggered',
    ['severity', 'source']
)
NOTIFICATION_LATENCY = Histogram(
    'notification_latency_seconds',
    'Notification processing latency'
)
ACTIVE_SUBSCRIBERS = Gauge(
    'notification_active_subscribers',
    'Number of active subscribers'
)
ANOMALY_SCORE = Gauge(
    'notification_anomaly_score',
    'Current anomaly score',
    ['service']
)
ALERT_QUEUE_SIZE = Gauge(
    'notification_queue_size',
    'Size of the alert queue'
)

# In-memory storage
alerts_history = deque(maxlen=MAX_ALERTS)
subscribers = {}
subscriber_id_counter = 0
sub_lock = threading.Lock()


class Alert:
    def __init__(self, alert_id, source, severity, message, metric_name=None,
                 metric_value=None, threshold=None, channel='log'):
        self.alert_id = alert_id
        self.source = source
        self.severity = severity  # critical, warning, info
        self.message = message
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.threshold = threshold
        self.channel = channel
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        self.status = 'active'

    def to_dict(self):
        return {
            'alert_id': self.alert_id,
            'source': self.source,
            'severity': self.severity,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'channel': self.channel,
            'timestamp': self.timestamp,
            'status': self.status
        }


def prune_old_alerts():
    cutoff = datetime.now().timestamp() - ALERT_RETENTION_SECONDS
    pruned = 0
    while alerts_history and alerts_history[0].timestamp and \
            datetime.fromisoformat(alerts_history[0].timestamp.replace('Z', '+00:00')).timestamp() < cutoff:
        alerts_history.popleft()
        pruned += 1
    if pruned > 0:
        logger.info(f"Pruned {pruned} old alerts")


@app.route('/')
def index():
    return jsonify({
        'service': 'Notification Service',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'dashboard': '/dashboard',
        'endpoints': {
            'health': '/health',
            'metrics': '/metrics',
            'send_alert': 'POST /alerts',
            'get_alerts': 'GET /alerts',
            'subscribe': 'POST /subscribe',
            'unsubscribe': 'DELETE /subscribe/<id>',
            'anomaly_report': 'POST /anomaly',
            'stats': 'GET /stats'
        }
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'notification-service',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'alerts_stored': len(alerts_history)
    })


@app.route('/metrics')
def metrics():
    ALERT_QUEUE_SIZE.set(len(alerts_history))
    ACTIVE_SUBSCRIBERS.set(len(subscribers))
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/alerts', methods=['POST'])
def send_alert():
    start_time = time.time()
    data = request.get_json() or {}

    severity = data.get('severity', 'warning')
    source = data.get('source', 'unknown')
    message = data.get('message', 'No message provided')
    metric_name = data.get('metric_name')
    metric_value = data.get('metric_value')
    threshold = data.get('threshold')
    channel = data.get('channel', 'log')

    alert_id = f"alert-{int(time.time()*1000)}"
    alert = Alert(
        alert_id=alert_id,
        source=source,
        severity=severity,
        message=message,
        metric_name=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        channel=channel
    )

    alerts_history.append(alert)

    ALERTS_SENT.labels(severity=severity, channel=channel).inc()
    ALERTS_TRIGGERED.labels(severity=severity, source=source).inc()

    if metric_name and source:
        ANOMALY_SCORE.labels(service=source).set(float(metric_value) if metric_value else 0)

    prune_old_alerts()

    latency = time.time() - start_time
    NOTIFICATION_LATENCY.observe(latency)

    logger.info(f"[{severity.upper()}] Alert from {source}: {message}")

    if channel == 'webhook' and subscribers:
        for sid, sub in list(subscribers.items()):
            if sub['severity_filter'] in ('all', severity):
                logger.info(f"Notifying subscriber {sid} via webhook")

    return jsonify({
        'status': 'sent',
        'alert_id': alert_id,
        'timestamp': alert.timestamp,
        'severity': severity
    }), 201


@app.route('/alerts', methods=['GET'])
def get_alerts():
    severity_filter = request.args.get('severity')
    source_filter = request.args.get('source')
    limit = int(request.args.get('limit', 50))

    filtered = []
    for alert in reversed(alerts_history):
        if severity_filter and alert.severity != severity_filter:
            continue
        if source_filter and alert.source != source_filter:
            continue
        filtered.append(alert.to_dict())
        if len(filtered) >= limit:
            break

    return jsonify({
        'count': len(filtered),
        'alerts': filtered
    })


@app.route('/anomaly', methods=['POST'])
def report_anomaly():
    data = request.get_json() or {}
    service = data.get('service', 'unknown')
    score = float(data.get('score', 0))
    details = data.get('details', {})

    ANOMALY_SCORE.labels(service=service).set(score)

    severity = 'critical' if score > 0.8 else 'warning' if score > 0.5 else 'info'

    alert_id = f"anomaly-{service}-{int(time.time()*1000)}"
    alert = Alert(
        alert_id=alert_id,
        source=f"anomaly-detector:{service}",
        severity=severity,
        message=f"Anomaly detected in {service} with score {score:.4f}",
        metric_name=details.get('metric_name', 'anomaly_score'),
        metric_value=score,
        threshold=details.get('threshold', 0.5),
        channel='log'
    )

    alerts_history.append(alert)
    ALERTS_TRIGGERED.labels(severity=severity, source=f'anomaly-detector:{service}').inc()
    ALERTS_SENT.labels(severity=severity, channel='log').inc()

    logger.info(f"Anomaly report: {service} score={score:.4f} severity={severity}")

    return jsonify({
        'status': 'processed',
        'alert_id': alert_id,
        'severity': severity,
        'timestamp': alert.timestamp
    }), 201


@app.route('/subscribe', methods=['POST'])
def subscribe():
    global subscriber_id_counter
    data = request.get_json() or {}

    with sub_lock:
        sid = subscriber_id_counter
        subscriber_id_counter += 1
        subscribers[sid] = {
            'endpoint': data.get('endpoint', 'log'),
            'severity_filter': data.get('severity_filter', 'all'),
            'created_at': datetime.utcnow().isoformat() + 'Z'
        }

    ACTIVE_SUBSCRIBERS.set(len(subscribers))
    logger.info(f"Subscriber {sid} registered: {subscribers[sid]}")

    return jsonify({'subscriber_id': sid, 'status': 'subscribed'}), 201


@app.route('/subscribe/<int:sid>', methods=['DELETE'])
def unsubscribe(sid):
    with sub_lock:
        if sid in subscribers:
            del subscribers[sid]
            ACTIVE_SUBSCRIBERS.set(len(subscribers))
            logger.info(f"Subscriber {sid} removed")
            return jsonify({'status': 'unsubscribed'})
    return jsonify({'error': 'Subscriber not found'}), 404


@app.route('/stats', methods=['GET'])
def stats():
    severity_counts = {'critical': 0, 'warning': 0, 'info': 0}
    source_counts = {}

    for alert in alerts_history:
        severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        source_counts[alert.source] = source_counts.get(alert.source, 0) + 1

    return jsonify({
        'total_alerts': len(alerts_history),
        'active_subscribers': len(subscribers),
        'severity_breakdown': severity_counts,
        'source_breakdown': source_counts,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Notification Service Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }
  header { background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%); padding: 24px 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
  h1 { margin: 0 0 6px 0; font-size: 26px; }
  .sub { color: #cbd5e1; font-size: 14px; }
  main { padding: 24px 32px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; border-radius: 10px; padding: 20px; border-left: 4px solid #6366f1; box-shadow: 0 1px 4px rgba(0,0,0,0.3); }
  .card.crit { border-left-color: #ef4444; }
  .card.warn { border-left-color: #f59e0b; }
  .card.info { border-left-color: #3b82f6; }
  .card .label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: 8px; }
  .card .value { font-size: 32px; font-weight: 700; }
  .section { background: #1e293b; border-radius: 10px; padding: 20px; margin-bottom: 24px; }
  .section h2 { margin-top: 0; font-size: 18px; border-bottom: 1px solid #334155; padding-bottom: 10px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #334155; font-size: 14px; }
  th { color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
  .badge.critical { background: #7f1d1d; color: #fecaca; }
  .badge.warning  { background: #78350f; color: #fde68a; }
  .badge.info     { background: #1e3a8a; color: #bfdbfe; }
  .bar { background: #334155; height: 8px; border-radius: 4px; overflow: hidden; }
  .bar > div { height: 100%; background: linear-gradient(90deg, #6366f1, #8b5cf6); }
  .row { display: flex; justify-content: space-between; align-items: center; margin: 8px 0; }
  .row .src { font-size: 13px; color: #cbd5e1; }
  .row .count { font-weight: 600; }
  .actions a { display: inline-block; margin-right: 12px; color: #93c5fd; text-decoration: none; font-size: 13px; }
  .actions a:hover { text-decoration: underline; }
  .refresh { float: right; font-size: 12px; color: #94a3b8; }
</style>
</head>
<body>
<header>
  <h1>Notification Service Dashboard</h1>
  <div class="sub">Online Boutique AIOps Extension · Real-time Alert Monitoring · <span id="ts"></span></div>
  <div class="actions" style="margin-top:10px;">
    <a href="/health">/health</a>
    <a href="/alerts">/alerts</a>
    <a href="/stats">/stats</a>
    <a href="/metrics">/metrics</a>
  </div>
</header>
<main>
  <div class="grid">
    <div class="card"><div class="label">Total Alerts</div><div class="value" id="total">-</div></div>
    <div class="card crit"><div class="label">Critical</div><div class="value" id="crit">-</div></div>
    <div class="card warn"><div class="label">Warning</div><div class="value" id="warn">-</div></div>
    <div class="card info"><div class="label">Info</div><div class="value" id="info">-</div></div>
  </div>
  <div class="section">
    <h2>Sources <span class="refresh">live</span></h2>
    <div id="sources"></div>
  </div>
  <div class="section">
    <h2>Recent Alerts <span class="refresh">latest 20</span></h2>
    <table>
      <thead><tr><th>Time (UTC)</th><th>Severity</th><th>Source</th><th>Message</th><th>Metric</th><th>Value</th><th>Threshold</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</main>
<script>
async function load() {
  try {
    const [stats, alerts] = await Promise.all([fetch('/stats').then(r=>r.json()), fetch('/alerts?limit=20').then(r=>r.json())]);
    document.getElementById('total').textContent = stats.total_alerts;
    document.getElementById('crit').textContent = (stats.severity_breakdown.critical||0);
    document.getElementById('warn').textContent = (stats.severity_breakdown.warning||0);
    document.getElementById('info').textContent = (stats.severity_breakdown.info||0);
    const sources = Object.entries(stats.source_breakdown).sort((a,b)=>b[1]-a[1]);
    const max = Math.max(1, ...sources.map(s=>s[1]));
    document.getElementById('sources').innerHTML = sources.map(([s,c])=>
      `<div class="row"><span class="src">${s}</span><span class="count">${c}</span></div>
       <div class="bar"><div style="width:${c/max*100}%"></div></div>`).join('') || '<em style="color:#64748b">No alerts yet</em>';
    document.getElementById('rows').innerHTML = alerts.alerts.map(a=>{
      const t = a.timestamp ? a.timestamp.replace('T',' ').replace('Z','') : '-';
      return `<tr>
        <td style="font-family:monospace;font-size:12px;color:#94a3b8">${t}</td>
        <td><span class="badge ${a.severity}">${a.severity}</span></td>
        <td>${a.source}</td>
        <td>${a.message||''}</td>
        <td>${a.metric_name||''}</td>
        <td>${a.metric_value!==null&&a.metric_value!==undefined?a.metric_value:'-'}</td>
        <td>${a.threshold!==null&&a.threshold!==undefined?a.threshold:'-'}</td>
      </tr>`;
    }).join('') || '<tr><td colspan="7" style="text-align:center;color:#64748b">No alerts</td></tr>';
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
def dashboard():
    return Response(DASHBOARD_HTML, mimetype='text/html')


if __name__ == '__main__':
    logger.info(f"Notification Service starting on port {os.getenv('PORT', 8080)}")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
