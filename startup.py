import time
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import WebDriverException

URL = "http://localhost:5000"

def wait_for_server():
    while True:
        try:
            response = requests.get(URL)
            if response.status_code == 200:
                print("Server is up!")
                return
        except requests.ConnectionError:
            pass
        print("Waiting for server...")
        time.sleep(2)

def launch_browser():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--start-fullscreen")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    try:
        driver = webdriver.Chrome(options=options)
    except WebDriverException as e:
        print("Error starting ChromeDriver:", e)
        return None

    driver.get(URL)

    time.sleep(1)

    actions = ActionChains(driver)
    actions.send_keys(Keys.F11).perform()

    print("Launched browser in fullscreen mode.")
    return driver

if __name__ == "__main__":
    wait_for_server()
    driver = launch_browser()

    if driver:
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting and closing browser...")
            driver.quit()
