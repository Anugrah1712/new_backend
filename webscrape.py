from playwright.async_api import async_playwright
import os
import google.generativeai as genai


genai.configure(api_key="AIzaSyBNJvzSaKq26JHLLMSlIYaZAzOANtc8FCY")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

def create_table_prompt(content):
    return (
        "You are analyzing a web page with one or more interest rate tables related to Fixed Deposits (FDs). "
        "For EACH table in the content below:\n"
        "- Mention the table's heading/title or any label that identifies the table.\n"
        "- Interpret all rows and columns precisely.\n"
        "- Clearly explain what each column means.\n"
        "- Summarize each row’s interest rate for all payout options.\n"
        "- Highlight the highest available rate and the corresponding tenure/payout.\n"
        "- Do NOT compare across tables. Each table should be explained independently.\n"
        "- Include any eligibility criteria near the tables if applicable.\n\n"
        "Here is the content:\n\n" + content
    )

def create_faq_prompt(content):
    return (
        "From the content below, extract up to 20 Frequently Asked Questions (FAQs). "
        "Include both questions found in the content and logical questions a user might ask. Format:\n"
        "Q: <question>\nA: <answer>\n\n"
        "Content:\n\n" + content
    )

async def scrape_web_data(links):
    scraped_data = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-gpu"
            ]
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            locale="en-US",
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
            permissions=["geolocation"],
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1"
    }
)
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
            });
        """)

        for url in links:
            try:
                print(f"\n🌐 Scraping URL: {url}")
                page = await context.new_page()
                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(3000)

                full_text = await page.inner_text("body")
                print(f"✅ Extracted full page text (truncated):\n{full_text[:1000]}...\n")

                structured_tables = []
                tables = await page.query_selector_all("table")
                print(f"📊 Found {len(tables)} tables on the page.")

                for i, table in enumerate(tables, start=1):
                    rows = await table.query_selector_all("tr")
                    table_data = []
                    for row in rows:
                        columns = await row.query_selector_all("th, td")
                        column_text = [await column.inner_text() for column in columns if column]
                        if column_text:
                            table_data.append([cell.strip() for cell in column_text])
                    if table_data:
                        structured_table = f"\nTable {i}:\n" + "\n".join([", ".join(row) for row in table_data])
                        print(f"📋 Extracted Table {i}:\n{structured_table}\n")
                        structured_tables.append(structured_table)

                structured_table_text = "\n\n".join(structured_tables)
                combined_content = full_text + "\n\n" + structured_table_text

                table_prompt = create_table_prompt(combined_content)
                faq_prompt = create_faq_prompt(combined_content)

                print("🤖 Sending table prompt to Gemini...")
                table_response = model.generate_content(table_prompt)
                print(f"✅ Gemini Table Response:\n{table_response.text[:1000]}...\n")

                print("🤖 Sending FAQ prompt to Gemini...")
                faq_response = model.generate_content(faq_prompt)
                print(f"✅ Gemini FAQ Response:\n{faq_response.text[:1000]}...\n")

                scraped_data.append({
                    "url": url,
                    "table_analysis": table_response.text,
                    "faq_extraction": faq_response.text,
                    "raw_text": full_text,
                    "tables_raw": structured_table_text
                })

            except Exception as e:
                print(f"❌ Error scraping {url}: {e}")
                scraped_data.append({"url": url, "error": str(e)})

        await browser.close()
    return scraped_data