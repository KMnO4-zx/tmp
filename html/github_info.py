import requests
from datetime import datetime
import pprint

# 获取 GitHub 用户信息
def get_github_user_info(github_id):
    # GitHub API URL
    url = f"https://api.github.com/users/{github_id}"

    # 发起请求
    response = requests.get(url)

    if response.status_code == 200:
        user_info = response.json()
        return user_info
    else:
        print("无法获取该用户信息，可能是 GitHub 用户名错误")

# 获取用户仓库列表
def get_user_repositories(github_id):
    url = f"https://api.github.com/users/{github_id}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repo_data = response.json()
    else:
        raise Exception(f"无法获取仓库信息: {github_id}")
    
    

# 输入 GitHub 用户名
github_id = 'KMnO4-zx'
user_info = get_github_user_info(github_id)
repos = get_user_repositories(github_id)

