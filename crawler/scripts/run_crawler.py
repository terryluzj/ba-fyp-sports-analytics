import scrapy.crawler
from scrapy.utils.project import get_project_settings


class ProfileCrawlerProcess(scrapy.crawler.CrawlerProcess):

    CRAWLER_NAME = 'netkeiba'

    def __init__(self):
        super(ProfileCrawlerProcess, self).__init__(settings=get_project_settings())

    def start_crawl(self):
        self.crawl(ProfileCrawlerProcess.CRAWLER_NAME)
        self.start()


def run_crawler():
    process = ProfileCrawlerProcess()
    process.start_crawl()


if __name__ == '__main__':
    run_crawler()
