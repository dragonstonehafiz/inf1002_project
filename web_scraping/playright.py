from playwright.sync_api import sync_playwright
from selectolax.parser import HTMLParser
import pandas as pd


def get_page_html(page, asin, page_number):
    url = f"https://www.amazon.sg/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={page_number}"
    page.goto(url)
    html = HTMLParser(page.content())
    return html


def get_data_list_from_html(html):
    reviews = html.css("div[data-hook=review]")
    print(len(reviews))
    if len(reviews) == 0:
        return False
    output_list = []
    for review in reviews:
        title = review.css_first("span[data-hook=review-title").text().replace("\n", "").strip()
        rating = review.css_first("i[data-hook=cmps-review-star-rating]").text()
        review_body = review.css_first("span[data-hook=review-body]").text().replace("\n", "").strip()

        data_dict = {
            "title": title,
            "rating": rating,
            "review_body": review_body
        }
        output_list.append(data_dict)

    return output_list


def run():
    asin = "1789091373"
    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    page = browser.new_page()
    total_data = []
    page_num = 1
    while True:
        print(f"reading page {page_num}")
        reviews = get_page_html(page, asin, page_num)
        data = get_data_list_from_html(reviews, asin)
        if data is False:
            break
        total_data.extend(data)
        page_num += 1
    dataframe = pd.DataFrame(total_data)
    dataframe.to_csv("reviews.csv", index=False)


def main():
    run()


if __name__ == "__main__":
    main()

