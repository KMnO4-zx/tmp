import requests

def get_total_stars(username):
    total_stars = 0
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            print("Failed to retrieve data: ", response.status_code)
            break

        repos = response.json()
        if not repos:
            break

        for repo in repos:
            total_stars += repo['stargazers_count']

        page += 1

    return total_stars

def get_total_stars_for_org(org_name):
    total_stars = 0
    page = 1
    while True:
        url = f"https://api.github.com/orgs/{org_name}/repos?per_page=100&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            print("Failed to retrieve data: ", response.status_code)
            break

        repos = response.json()
        if not repos:
            break

        for repo in repos:
            total_stars += repo['stargazers_count']

        page += 1

    return total_stars

# 替换为目标 GitHub 用户名
username = "KMnO4-zx"
print(f"Total stars for {username}: {get_total_stars(username)}")

# 替换为目标 GitHub 组织名称
org_name = "datawhalechina"
internlm = 'InternLM'
openmmlab = 'open-mmlab'
print(f"Total stars for {org_name}: {get_total_stars_for_org(org_name)}")
print(f"Total stars for {openmmlab}: {get_total_stars_for_org(openmmlab)}")
print(f"Total stars for {internlm}: {get_total_stars_for_org(internlm)}")