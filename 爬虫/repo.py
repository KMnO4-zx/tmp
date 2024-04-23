import requests

# GitHub API URL for organization's repositories
org = "datawhalechina"
api_url = f"https://api.github.com/orgs/{org}/repos"
token = "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"  # 把这里替换成你的 GitHub 访问令牌

def count_stars(api_url, token):
    total_stars = 0
    headers = {"Authorization": f"token {token}"}
    while True:
        response = requests.get(api_url, headers=headers)
        repos = response.json()

        # Accumulate stars from each repository
        for repo in repos:
            total_stars += repo['stargazers_count']

        # Find the link to the next page of repositories, if it exists
        if 'next' in response.links:
            api_url = response.links['next']['url']
        else:
            break

    return total_stars

# Call the function and print the total number of stars
total_stars = count_stars(api_url, token)
print(f"Total stars across all repositories: {total_stars}")