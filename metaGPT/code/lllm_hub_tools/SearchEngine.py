import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from lllm_hub_tools.extend.GoogleTool import search

class SearchEngine:
    def __init__(self, engine_name="google"):
        self.engine_name = engine_name

    def search(self, query,max_results=20):
        list_dict = []
        if self.engine_name == "google":
            list_dict = self.google_search_by_new(query=query,max_results=max_results)
        return list_dict
    
    def google_search_by_new(self,query,max_results=20):
        result = search(query, stop=max_results)
        newresult = []
        for item in result:
            if item["link"] is not None:
                newresult.append(item)
        return newresult

