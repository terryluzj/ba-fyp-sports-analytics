import scrapy


class KeibaCrawler(scrapy.Spider):

    name = 'core'

    def __init__(self, *args, **kwargs):
        # Get faculty link for each university
        super(KeibaCrawler, self).__init__(*args, **kwargs)

    def start_requests(self):
        pass

    def parse(self, response):
        pass
