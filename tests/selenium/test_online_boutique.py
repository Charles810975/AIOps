import argparse
import csv
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait


def build_driver(browser, headless, driver_path=None, browser_binary=None):
    browser = browser.lower()
    if browser == "chrome":
        options = ChromeOptions()
        if browser_binary:
            options.binary_location = browser_binary
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1440,1000")
        service = ChromeService(executable_path=driver_path) if driver_path else None
        return webdriver.Chrome(service=service, options=options)
    if browser == "edge":
        options = EdgeOptions()
        if browser_binary:
            options.binary_location = browser_binary
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1440,1000")
        service = EdgeService(executable_path=driver_path) if driver_path else None
        return webdriver.Edge(service=service, options=options)
    raise ValueError(f"Unsupported browser: {browser}")


def measure(results, name, func, screenshot_dir, driver):
    start = time.perf_counter()
    ok = True
    error = ""
    try:
        func()
    except Exception as exc:
        ok = False
        error = str(exc)
    elapsed = time.perf_counter() - start
    screenshot = ""
    try:
        screenshot_path = screenshot_dir / f"{len(results) + 1:02d}_{name}.png"
        driver.save_screenshot(str(screenshot_path))
        screenshot = str(screenshot_path)
    except Exception:
        pass
    results.append({"step": name, "ok": ok, "elapsed_seconds": elapsed, "error": error, "screenshot": screenshot})
    return ok


def click_first(driver, selectors):
    for selector in selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        visible = [e for e in elements if e.is_displayed() and e.is_enabled()]
        if visible:
            visible[0].click()
            return
    raise RuntimeError(f"No clickable element found for selectors: {selectors}")


def click_by_text_or_selector(driver, texts, selectors):
    lowered = [text.lower() for text in texts]
    candidates = driver.find_elements(By.CSS_SELECTOR, "a, button, input[type='submit']")
    for element in candidates:
        if not element.is_displayed() or not element.is_enabled():
            continue
        value = " ".join([
            element.text or "",
            element.get_attribute("value") or "",
            element.get_attribute("aria-label") or "",
            element.get_attribute("href") or "",
        ]).lower()
        if any(text in value for text in lowered):
            element.click()
            return
    click_first(driver, selectors)


def type_if_present(driver, selector, value):
    elements = driver.find_elements(By.CSS_SELECTOR, selector)
    if elements:
        elements[0].clear()
        elements[0].send_keys(value)
        return True
    return False


def select_if_present(driver, selector, value=None, index=1):
    elements = driver.find_elements(By.CSS_SELECTOR, selector)
    if not elements:
        return False
    select = Select(elements[0])
    if value is not None:
        select.select_by_value(value)
    else:
        select.select_by_index(min(index, len(select.options) - 1))
    return True


def main():
    parser = argparse.ArgumentParser(description="Selenium functional test for Online Boutique")
    parser.add_argument("--url", required=True, help="Online Boutique frontend URL")
    parser.add_argument("--output", default="reports/selenium/selenium_results.csv")
    parser.add_argument("--browser", choices=["chrome", "edge"], default="chrome")
    parser.add_argument("--driver-path", default=None, help="Path to chromedriver.exe or msedgedriver.exe")
    parser.add_argument("--browser-binary", default=None, help="Path to browser executable if Selenium cannot find it")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    screenshot_dir = output.parent / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    driver = build_driver(args.browser, args.headless, args.driver_path, args.browser_binary)
    wait = WebDriverWait(driver, 25)
    results = []

    try:
        measure(results, "open_home", lambda: driver.get(args.url), screenshot_dir, driver)

        def browse_product():
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/product/']")))
            click_first(driver, ["a[href*='/product/']"])
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        measure(results, "browse_product_detail", browse_product, screenshot_dir, driver)

        def add_to_cart():
            click_first(driver, ["button[type='submit']", "button", "input[type='submit']"])
            time.sleep(1)

        measure(results, "add_product_to_cart", add_to_cart, screenshot_dir, driver)

        def open_cart():
            driver.get(args.url.rstrip("/") + "/cart")
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        measure(results, "open_cart", open_cart, screenshot_dir, driver)

        def checkout():
            click_by_text_or_selector(
                driver,
                ["checkout", "place order", "order", "purchase"],
                ["a[href*='checkout']", "button[type='submit']", "input[type='submit']", "button"],
            )
            time.sleep(1)
            type_if_present(driver, "input[name='email']", "test@example.com")
            type_if_present(driver, "input[name='street_address']", "1 Software Testing Road")
            type_if_present(driver, "input[name='zip_code']", "100000")
            type_if_present(driver, "input[name='city']", "Beijing")
            type_if_present(driver, "input[name='state']", "Beijing")
            type_if_present(driver, "input[name='country']", "China")
            type_if_present(driver, "input[name='credit_card_number']", "4111111111111111")
            type_if_present(driver, "input[name='credit_card_expiration_month']", "12")
            type_if_present(driver, "input[name='credit_card_expiration_year']", "2030")
            type_if_present(driver, "input[name='credit_card_cvv']", "123")
            select_if_present(driver, "select[name='currency_code']", index=1)
            submit_elements = driver.find_elements(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], button")
            if submit_elements:
                click_by_text_or_selector(
                    driver,
                    ["place order", "submit", "pay", "order", "purchase"],
                    ["button[type='submit']", "input[type='submit']", "button"],
                )
                time.sleep(2)

        measure(results, "checkout_or_submit_order", checkout, screenshot_dir, driver)
    finally:
        driver.quit()

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "ok", "elapsed_seconds", "error", "screenshot"])
        writer.writeheader()
        writer.writerows(results)

    total = sum(row["elapsed_seconds"] for row in results)
    passed = sum(1 for row in results if row["ok"])
    print(f"Selenium results saved to {output}")
    print(f"Passed {passed}/{len(results)} steps, total time {total:.3f}s")


if __name__ == "__main__":
    main()
