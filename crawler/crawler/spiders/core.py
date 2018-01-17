import datetime
import scrapy
from scrapy import Request


class KeibaCrawler(scrapy.Spider):

    name = 'core'
    custom_settings = {
        # Override custom settings preset by scrapy
        # Take a depth-first search algorithm by setting a negative priority or positive otherwise
        'DEPTH_LIMIT': 4,
        'DEPTH_PRIORITY': -2,
        'DEPTH_STATS_VERBOSE': True,

        # Limit the concurrent request per domain and moderate the server load
        'CONCURRENT_REQUESTS': 32,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'DOWNLOAD_DELAY': 0.5,
    }

    DOMAIN_URL = ['http://db.netkeiba.com']
    START_DATE = '06/01/2000'

    def __init__(self, *args, **kwargs):
        # Get faculty link for each university
        super(KeibaCrawler, self).__init__(*args, **kwargs)
        self.allowed_domains = list(KeibaCrawler.DOMAIN_URL)
        start_date = datetime.datetime.strptime(KeibaCrawler.START_DATE, '%d/%m/%Y').date()
        self.dates = []

        while start_date <= datetime.datetime.utcnow().date():
            if (start_date.weekday() == 5) | (start_date.weekday() == 6):
                self.dates.append(str(start_date))
            start_date += datetime.timedelta(days=1)

    def start_requests(self):
        for date in self.dates[:1]:
            request = Request('http://db.netkeiba.com/race/list/%s/' % date.replace('-', ''), callback=self.parse)
            request.meta['date'] = date
            yield request

    def parse(self, response):
        top_path = '//dl[@class[contains(., "race_top_hold_list")]]'
        sub_path_list = list(map(lambda path: top_path + '/%s' % path, [
            'dt/p/text()',  # Place
            'dd//dt/text()',  # Race Number
            '/dd//a/@title',  # Race Title
            '/dd//a/@href'  # Href
        ]))

        place_list = list(map(lambda name: name.strip(), response.xpath(sub_path_list[0]).extract()))
        race_num_list = response.xpath(sub_path_list[1]).extract()
        title_list = response.xpath(sub_path_list[2]).extract()
        href_list = response.xpath(sub_path_list[3]).extract()

        race_list = [[int(race_num.strip('R')), title, link] for race_num, title, link in zip(race_num_list,
                                                                                              title_list,
                                                                                              href_list)]
        link_dict = {}

        current_idx = 0
        current_number = 0
        for element in race_list:
            if element[0] <= current_number:
                current_idx += 1
            current_number = element[0]
            link_dict[' '.join([response.meta['date'],
                                place_list[current_idx],
                                '%sR' % element[0],
                                element[1]])] = element[2]

