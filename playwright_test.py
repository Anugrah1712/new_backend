import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = await browser.new_page()
        await page.goto("https://www.bajajfinserv.in/investments/fixed-deposit-application-form")
        print(await page.title())
        await browser.close()

asyncio.run(run())
