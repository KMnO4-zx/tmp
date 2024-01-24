# GitHub API 封装类文档

## 概览

这份文档提供了`GithubApi`类的概览和使用指南，这是一个用于与GitHub交互的Python封装类。该类允许您执行如获取仓库的README、检索仓库统计数据、获取最新提交和解析GitHub趋势仓库等操作。

## 安装

在使用`GithubApi`类之前，您需要在系统上安装Python以及`requests`和`BeautifulSoup`库。您可以使用`pip`来安装这些库：

```bash
pip install requests beautifulsoup4
```

## 类方法

`GithubApi`类提供以下方法：

### `get_readme(user, repo, max_len=10000)`

获取指定用户和仓库的README文件。

参数：
- `user`: GitHub用户名。
- `repo`: GitHub仓库名。
- `max_len`: 可选，指定返回的README文本的最大长度，默认为10000字符。

返回：
- 成功时返回README文本的字符串，否则返回空字符串。

### `get_repo_stats(user, repo)`

获取指定用户和仓库的统计数据。

参数：
- `user`: GitHub用户名。
- `repo`: GitHub仓库名。

返回：
- 成功时返回包含星标数和分支数的字典，否则返回`None`。
- 
### `get_repo_star_history(user, repo)`

获取指定用户和仓库的全部star数据。

参数：
- `user`: GitHub用户名。
- `repo`: GitHub仓库名。

返回：
- 成功时返回包含日期，当天星标数，总星标数的字典列表，否则返回`None`。
### `get_latest_commit(user, repo, max_num=3000)`

获取指定用户和仓库的最新提交。

参数：
- `user`: GitHub用户名。
- `repo`: GitHub仓库名。
- `max_num`: 可选，指定返回的提交数量的最大值，默认为3000。

返回：
- 成功时返回一个包含提交信息的列表，否则返回`None`。

### `fetch_html(url)`

获取指定URL的HTML内容。

参数：
- `url`: 要获取HTML的网页地址。

返回：
- 返回网页的HTML内容字符串。

### `parse_github_trending()`

解析GitHub趋势页面，获取当前趋势的仓库信息。

返回：
- 返回一个列表，包含了解析出的仓库信息字典。

## 使用示例

以下是`GithubApi`类的一些使用示例：

```python
# 初始化GithubApi类
github_api = GithubApi()

# 获取指定仓库的README
readme_content = github_api.get_readme('octocat', 'Hello-World')
print(readme_content)

# 获取指定仓库的统计数据
repo_stats = github_api.get_repo_stats('octocat', 'Hello-World')
print(repo_stats)

# 获取指定仓库的最新提交
latest_commits = github_api.get_latest_commit('octocat', 'Hello-World')
for commit in latest_commits:
    print(commit)

# 解析GitHub趋势页面
trending_repositories = github_api.parse_github_trending()
for repo in trending_repositories:
    print(repo)
```

请确保在使用这些方法之前，您已经正确安装了所有必需的依赖项，并且您的网络环境可以正常访问GitHub。