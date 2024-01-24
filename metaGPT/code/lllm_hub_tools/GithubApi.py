import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from datetime import datetime
class GithubApi:
    def __init__(self):
        pass

    def get_readme(self , user, repo,max_len=10000):
        url = f"https://raw.githubusercontent.com/{user}/{repo}/master/README.md"
        response = requests.get(url)
        if response.status_code == 200:
            #print(response.text[:max_len])
            if len(response.text) > max_len:
                return response.text[:max_len]
            else:
                return response.text
        else:
            return ""

    def get_repo_stats(self , user, repo):
        url = f"https://api.github.com/repos/{user}/{repo}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            retjson = {}
            retjson["stars"] = data['stargazers_count']
            retjson["forks"] = data['forks_count']
            return retjson
        else:
            return None

    def get_repo_star_history(self, user, repo,show_type=1):
        url = f"https://143.47.235.108:8090/allStars?repo={user}/{repo}"
        ua = UserAgent()
        # 准备自定义的headers信息
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive',
            'Host': '143.47.235.108:8090',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': ua.random,  # 使用fake_useragent生成的随机User-Agent
            'sec-ch-ua': '"Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?1',
            'sec-ch-ua-platform': '"Android"'
        }
        response = requests.get(url,headers=headers,verify=False)
        #print(url)
        #print(response)
        if response.status_code == 200:
            data = response.json()
            ret_list = []
            if show_type == 2:
                for item in data:
                    date_obj = datetime.strptime(item[0], "%d-%m-%Y")
                    item[0] = date_obj.strftime("%Y-%m-%d")
                    tmp_dict = {"date": item[0],"today_stars": item[0], "total_stars": item[1] }
                    ret_list.append(tmp_dict)
                return ret_list
            else:
                return data
        else:
            return None

    def get_latest_commit(self,user, repo,max_num=3000):
        url = f"https://api.github.com/repos/{user}/{repo}/commits"
        response = requests.get(url)
        if response.status_code == 200:
            commits = response.json()
            deal_commits = commits
            if len(commits) > max_num:
                deal_commits = commits[:max_num]
            ret_list = []
            for commit in deal_commits:
                one_commit = {}
                one_commit['sha'] = commit['sha']
                one_commit['node_id'] = commit['node_id']
                one_commit['commit_author'] = commit['commit']['author']
                one_commit['commit_message']  = commit['commit']['message']
                one_commit['commit_url'] = commit['html_url']
                one_commit['commit_date'] = commit['commit']['committer']['date']
                ret_list.append(one_commit)
            return ret_list
        else:
            return None

    def fetch_html(self,url):
        response = requests.get(url)
        return response.text

    def parse_github_trending(self):
        url = 'https://github.com/trending'
        response = self.fetch_html(url)
        soup = BeautifulSoup(response, 'html.parser')
        repositories = []
        for article in soup.select('article.Box-row'):
            repo_info = {}
            repo_info['name'] = article.select_one('h2 a').text.strip()
            repo_info['url'] = article.select_one('h2 a')['href'].strip()
            # Description
            description_element = article.select_one('p')
            repo_info['description'] = description_element.text.strip() if description_element else None
            # Language
            language_element = article.select_one('span[itemprop="programmingLanguage"]')
            repo_info['language'] = language_element.text.strip() if language_element else None
            # Stars and Forks
            stars_element = article.select('a.Link--muted')[0]
            forks_element = article.select('a.Link--muted')[1]
            repo_info['stars'] = stars_element.text.strip()
            repo_info['forks'] = forks_element.text.strip()
            # Today's Stars
            today_stars_element = article.select_one('span.d-inline-block.float-sm-right')
            repo_info['today_stars'] = today_stars_element.text.strip() if today_stars_element else None
            repositories.append(repo_info)

            for repo in repositories:
                repo['name'] = repo['name'].replace('\n', '')
                repo['name'] = repo['name'].replace(' ', '')

        return repositories