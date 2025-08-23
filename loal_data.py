import time
import json
import re
import pandas as pd
from urllib.parse import urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, 
    ElementClickInterceptedException, StaleElementReferenceException
)

csv_file_path = "new_york_weather_2024_1-6.csv"
is_first_write = not pd.io.common.file_exists(csv_file_path)
def clean_time(raw_time):
    if not raw_time:
        return ""
    cleaned = raw_time.replace('\n', ' ').strip()
    match_24h = re.match(r'^\d{1,2}:\d{2}', cleaned)
    match_12h = re.match(r'^\d{1,2}:\d{2} (AM|PM)', cleaned, re.IGNORECASE)
    if match_24h:
        return match_24h.group(0)
    elif match_12h:
        return match_12h.group(0)
    else:
        print(f"时间格式未匹配：{cleaned}")
        return cleaned

def merge_datetime(date_str, cleaned_time):
    if not date_str or not cleaned_time:
        return ""
    return f"{date_str} {cleaned_time}"
def append_to_csv(data_list):
    global is_first_write
    if not data_list:
        return
    df = pd.DataFrame(data_list)
    df['row_index'] = df.apply(
        lambda x: merge_datetime(x['date'], x['time']),
        axis=1
    )
    df.set_index('row_index', inplace=True)
    df.to_csv(
        csv_file_path,
        mode='a',
        header=is_first_write,
        encoding='utf-8'
    )
    if is_first_write:
        is_first_write = False

def crawl_month_weather(month):
    # 月份页面URL
    month_url = f"https://www.timeanddate.com/weather/usa/new-york/historic?month={month}&year=2024"
    print(f"\n{'='*50}")
    print(f"开始爬取2024年{month}月数据，页面URL: {month_url}")
    
    option = ChromeOptions()
    option.add_argument("--disable-blink-features=AutomationControlled")
    option.add_experimental_option('detach', True)
    option.add_argument('--incognito')
    option.add_argument('--ignore-certificate-errors')
    option.add_argument('--ignore-ssl-errors')
    option.add_experimental_option('excludeSwitches', ['enable-automation'])
    option.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    driver = webdriver.Chrome(options=option)
    driver.get(month_url)
    driver.maximize_window()
    time.sleep(3)  
    
    try:
        print(f"2024年{month}月：等待日期容器加载...")
        weather_links_divs = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.XPATH, "//div[@class='weatherLinks']"))
        )
        weather_links_div = weather_links_divs[1] if len(weather_links_divs) >= 2 else weather_links_divs[0]
        
        date_elements = weather_links_div.find_elements(By.TAG_NAME, 'a')
        valid_date_elements = [
            elem for elem in date_elements 
            if elem.get_attribute('href') and 'hd=' in elem.get_attribute('href')
        ]
        
        if not valid_date_elements:
            print(f"2024年{month}月：未找到有效日期链接，跳过该月份")
            driver.quit()
            return
        
        print(f"2024年{month}月：共找到 {len(valid_date_elements)} 天数据，开始爬取...")
        
        for index, date_elem in enumerate(valid_date_elements):
            try:
                date_text = date_elem.text.strip()
                href = date_elem.get_attribute('href')
                hd_param = parse_qs(urlparse(href).query).get('hd', [None])[0]
                
                if hd_param and len(hd_param) == 8:
                    date_str = f"{hd_param[:4]}-{hd_param[4:6]}-{hd_param[6:8]}"
                else:
                    date_str = date_text
                    print(f"2024年{month}月：日期解析失败，使用原始文本 {date_text}")
                
                print(f"\n[{index+1}/{len(valid_date_elements)}] 处理日期：{date_str}")

                try:
                    WebDriverWait(driver, 5).until(EC.element_to_be_clickable(date_elem))
                    driver.execute_script("arguments[0].scrollIntoView(true);", date_elem)
                    time.sleep(0.5)
                    date_elem.click()
                except (ElementClickInterceptedException, StaleElementReferenceException):
                    print("元素状态异常，尝试重新定位并点击...")
                    date_elem = driver.find_element(By.XPATH, f"//a[@href='{href}']")
                    driver.execute_script("arguments[0].click();", date_elem)
                time.sleep(2)

                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'wt-his'))
                )

                table = driver.find_element(By.ID, 'wt-his')
                tbody = table.find_element(By.TAG_NAME, 'tbody')
                rows = tbody.find_elements(By.TAG_NAME, 'tr')
                
                daily_data = []  
                for tr in rows:
                    try:
                        th_element = tr.find_element(By.TAG_NAME, 'th')
                        if th_element.text.strip() == 'Time':
                            continue
                    except NoSuchElementException:
                        continue

                    data_dict = {
                        'date': date_str,
                        'time': None,
                        'temp': None,
                        'weather': None,
                        'wind': None,
                        'humidity': None,
                        'barometer': None,
                        'visibility': None
                    }
                    
                    try:
                        raw_time = th_element.text.strip()
                        data_dict['time'] = clean_time(raw_time) 
                    except NoSuchElementException:
                        pass

                    try:
                        all_td = tr.find_elements(By.TAG_NAME, 'td')
                        if len(all_td) >= 8:
                            data_dict['temp'] = all_td[1].text.strip()
                            data_dict['weather'] = all_td[2].text.strip()
                            data_dict['wind'] = all_td[3].text.strip()
                            data_dict['humidity'] = all_td[5].text.strip()
                            data_dict['barometer'] = all_td[6].text.strip()
                            data_dict['visibility'] = all_td[7].text.strip()
                    except (NoSuchElementException, IndexError) as e:
                        print(f"小时数据提取错误: {e}")
                        continue

                    if data_dict.get('time') and data_dict.get('temp'):
                        daily_data.append(data_dict)

                if daily_data:
                    append_to_csv(daily_data)
                    print(f"{date_str} 数据处理完成，共 {len(daily_data)} 条记录，已追加到CSV")
                else:
                    print(f"{date_str} 未提取到有效数据")

            except TimeoutException:
                print(f"{date_text} 页面加载超时，跳过该日期")
                continue
            except Exception as e:
                print(f"处理 {date_text} 时出错: {e}，继续下一个日期")
                continue

        print(f"\n2024年{month}月数据爬取完成！")

    except TimeoutException:
        print(f"2024年{month}月：日期容器加载超时，跳过该月份")
    except Exception as e:
        print(f"2024年{month}月爬取失败: {e}")
    finally:
        driver.quit()  

if __name__ == "__main__":
    for month in range(1, 7):
        crawl_month_weather(month)
    
    print(f"\n{'='*50}")
    print("所有月份数据爬取完成！")
    print(f"数据已保存到：{csv_file_path}")
    df = pd.read_csv(csv_file_path)
    print(f"总记录数：{len(df)} 条")