import datetime
import scrapy
from lxml import html
from scrapy import Request


class NetKeibaCrawler(scrapy.Spider):

    # This class targets at crawling race records from Netkeiba by first storing race link and subsequently
    # yield race record items as defined in items.py

    name = 'netkeiba'
    custom_settings = {
        # Override custom settings preset by scrapy
        # Take a depth-first search algorithm by setting a negative priority or positive otherwise
        'DEPTH_LIMIT': 4,
        'DEPTH_PRIORITY': 2,
        'DEPTH_STATS_VERBOSE': True,

        # Limit the concurrent request per domain and moderate the server load
        'CONCURRENT_REQUESTS': 32,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'DOWNLOAD_DELAY': 1,
    }

    DOMAIN_URL = ['db.netkeiba.com']

    # Start from the earliest available date
    # TODO: Scan through obtained data and update start date accordingly
    START_DATE = '06/01/2000'

    def __init__(self, *args, **kwargs):
        # Get faculty link for each university
        super(NetKeibaCrawler, self).__init__(*args, **kwargs)
        self.allowed_domains = list(NetKeibaCrawler.DOMAIN_URL)
        start_date = datetime.datetime.strptime(NetKeibaCrawler.START_DATE, '%d/%m/%Y').date()
        self.dates = []

        # National events are only held on weekend and hence filter out irrelevant date
        while start_date <= datetime.datetime.utcnow().date():
            if (start_date.weekday() == 5) | (start_date.weekday() == 6):
                self.dates.append(str(start_date))
            start_date += datetime.timedelta(days=1)

    def start_requests(self):
        # Add indexing for debugging
        for date in self.dates[:1]:
            # Yielding request and provide relevant meta data
            request = Request('http://db.netkeiba.com/race/list/%s/' % date.replace('-', ''), callback=self.parse)
            request.meta['date'] = date
            yield request

    def parse(self, response):
        # Parse page content at the top level

        # Define XPath to extract information from the page
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

        # Re-organize information order and store into a temporary dictionary for next-level parsing
        race_list = [[int(race_num.strip('R')), title, link] for race_num, title, link in zip(race_num_list,
                                                                                              title_list,
                                                                                              href_list)]
        link_dict = {}

        # Linearly scan through the race record list
        current_idx = 0
        current_number = 0

        # Add indexing for debugging
        for element in race_list[:1]:
            if element[0] <= current_number:
                current_idx += 1
            current_number = element[0]
            link_dict[' '.join([response.meta['date'],
                                place_list[current_idx],
                                '%sR' % element[0],
                                element[1]])] = element[2]

        # Iterate through link dict and yield next-level request
        for key, value in link_dict.items():
            # Initiate new meta data
            meta_list = key.split(' ')
            target_meta = {
                'custom': {
                    'date': meta_list[0],
                    'place': meta_list[1],
                    'race': meta_list[2],
                    'title': meta_list[3]
                }
            }
            yield response.follow(value, meta=target_meta, callback=self.parse_race)

    def parse_race(self, response):
        # Get basic information of the current race record page
        info_content = response.xpath('//diary_snap_cut/span/text()[normalize-space(.)]').extract_first().split('/')
        info_content = list(map(lambda text: text.strip(), info_content))
        basic_info = [
            info_content[0][0],  # Course Type
            info_content[0][1],  # Race Direction
            info_content[0][2:],  # Run Distance
            info_content[1].split(' : ')[-1],  # Weather
            info_content[2].split(' : ')[-1],  # Condition
            info_content[3].split(' : ')[-1]  # Start Time
        ]

        # Get the table content from the current page
        table_content = self.get_table_rows(response)
        row_content = table_content[0]
        row_data = table_content[1]
        row_link = list(map(lambda content: list(map(lambda anchor: anchor.attrib['href'],
                                                     content.xpath('//td//a[@href]'))), row_content))

        # Iterate through each row element
        for row_element, link_element in zip(row_data, row_link):
            try:
                assert (len(row_element) == 17) | (len(row_element) == 18)
                if row_element[0] == '1':
                    row_element = row_element[:8] + ['-'] + row_element[8:]
                try:
                    row_element[-1] = str(float(row_element[-1].replace(',', '')))
                except ValueError:
                    row_element.append('-')
                row_element[-4] = row_element[-4].strip()
            except AssertionError:
                self.logger.info('Incomplete record found at %s: %s' % (response.url, ' '.join(row_element).strip()))
            # Concatenate trainer name
            row_element[-3] = row_element[-4] + row_element[-3]
            del row_element[-4]
            curr_record = {
                'record': list(response.meta['custom'].values()) + basic_info + row_element
            }

            # Yield next-level request for horse, jockey, owner and trainer
            for link in sorted(link_element):
                if 'horse' in link:
                    yield response.follow(link, callback=self.parse_horse, meta=curr_record)
                elif 'jockey' in link:
                    yield response.follow(self.format_link(link, 'result'),
                                          callback=self.parse_jockey, meta=curr_record)
                elif 'owner' in link:
                    yield response.follow(self.format_link(link, 'result'), callback=self.parse_owner, meta=curr_record)
                elif 'trainer' in link:
                    yield response.follow(self.format_link(link, 'result'),
                                          callback=self.parse_trainer, meta=curr_record)

    def parse_horse_breed(self, response):
        # Intermediary step for parent horse information crawling
        content = list(map(lambda text: html.fromstring(text), response.xpath('//td[@rowspan="16"]').extract()))
        link_list = [None if len(element.xpath('a/@href')) <= 0 else element.xpath('a/@href')[0] for element in content]
        for link in link_list:
            yield response.follow(link, callback=self.parse_horse, meta=response.meta)

    def parse_horse(self, response):
        # Extract basic information
        basic_info = response.xpath('//p[@class="txt_01"]/text()').extract_first().split(u'\u3000')
        if len(basic_info) == 2:
            basic_info = ['-'] + basic_info
        elif len(basic_info) == 1:
            basic_info = ['-'] + basic_info + ['-']
        info_dict = {
            'status': basic_info[0],
            'gender': basic_info[1],
            'breed': basic_info[2]
        }

        # Extract profile of the current horse page
        profile = response.xpath('//table[@class[contains(., "db_prof_table")]]//tr').extract()
        profile_read = list(map(lambda text: html.fromstring(text), profile))
        profile_zipped = list(map(lambda content: content.xpath('//text()[normalize-space(.)]'), profile_read))
        profile_zipped = list(map(lambda x: [x[0], ' '.join(x[1:]).strip()], profile_zipped))
        profile_dict = {item[0]: item[1] for item in profile_zipped}
        profile_dict.update(info_dict)

        # Extract breeder information
        breeder_link = response.xpath('//a[@href[contains(., "breeder/")]]/@href').extract_first()
        breeder_meta = response.meta.copy()
        breeder_meta['breeder_name'] = profile_dict[u'生産者']
        if breeder_link is not None:
            yield response.follow(self.format_link(breeder_link, 'result'),
                                  callback=self.parse_breeder, meta=breeder_meta)

        # Get parent information
        if not response.meta.get('parent', False):
            parent = list(map(lambda text: html.fromstring(text), response.xpath('//td[@rowspan="2"]').extract()))
            parent = {' '.join(element.xpath('//text()[normalize-space(.)]')): '-'
                      if len(element.xpath('//a/@href')) <= 0 else element.xpath('//a/@href')[0]
                      for element in parent}
            for key, value in parent.items():
                response.meta['depth'] = -1
                new_meta = response.meta.copy()
                new_meta['depth'] = response.meta['depth']
                new_meta['parent'] = True
                yield response.follow(value, callback=self.parse_horse_breed, meta=new_meta)

    def parse_breeder(self, response):
        # Get table content and basic information
        row_data = self.get_table_rows(response)[1][2:]
        breeder_name = response.meta['breeder_name']
        for row_element in row_data:
            print([breeder_name] + row_element)

    def parse_owner(self, response):
        # Get table content and basic information
        row_data = self.get_table_rows(response)[1][2:]
        owner_name = response.meta['record'][-2]
        for row_element in row_data:
            print([owner_name] + row_element)

    def parse_jockey(self, response):
        # Get table content and basic information
        row_data = self.get_table_rows(response)[1][2:]
        jockey_name = response.meta['record'][16]
        basic_info = response.xpath('//p[@class="txt_01"]/text()[normalize-space(.)]').extract_first()
        try:
            basic_info = basic_info.strip().replace('\n', ' ').split(' ')[0]
        except TypeError:
            basic_info = ''
        except IndexError:
            basic_info = ''

        # Yield new request to parse profile page
        profile_link = 'http://db.netkeiba.com/jockey/profile/%s/' % response.url.split('/')[-2]
        new_meta = {
            'row_data': row_data,
            'jockey_name': jockey_name,
            'basic_info': basic_info,
        }
        yield Request(profile_link, callback=self.parse_jockey_profile, meta=new_meta)

    def parse_trainer(self, response):
        # Get table content and basic information
        row_data = self.get_table_rows(response)[1][2:]
        trainer_name = response.meta['record'][-3]
        basic_info = response.xpath('//p[@class="txt_01"]/text()[normalize-space(.)]').extract_first()
        try:
            basic_info = basic_info.strip().replace('\n', ' ').split(' ')[0]
        except TypeError:
            basic_info = ''
        except IndexError:
            basic_info = ''

        # Yield new request to parse profile page
        profile_link = 'http://db.netkeiba.com/trainer/profile/%s/' % response.url.split('/')[-2]
        new_meta = {
            'row_data': row_data,
            'jockey_name': trainer_name,
            'basic_info': basic_info,
        }
        yield Request(profile_link, callback=self.parse_trainer_profile, meta=new_meta)

    def parse_jockey_profile(self, response):
        # Get profile information
        profile_info = self.get_profile_table(response)
        print(profile_info)

    def parse_trainer_profile(self, response):
        # Get profile information
        profile_info = self.get_profile_table(response)
        print(profile_info)

    @staticmethod
    def get_table_rows(response):
        # Get table content by the following XPath
        table_content = response.xpath('//table[@class[contains(., "race_table")]]/tr')
        row_content = list(map(lambda content: html.fromstring(content), table_content.extract()))[1:]
        row_data = list(map(lambda html_content: html_content.xpath('//td//text()[normalize-space(.)]'), row_content))
        return row_content, row_data

    @staticmethod
    def format_link(link_string, append_string):
        # Format link to get extra information
        link_element = link_string.split('/')
        link_element.insert(2, append_string)
        return '/'.join(link_element)

    @staticmethod
    def get_profile_table(response):
        profile = response.xpath('//tr').extract()
        profile_read = list(map(lambda text: html.fromstring(text), profile))
        profile_zipped = list(map(lambda content: content.xpath("//text()[normalize-space(.)]"), profile_read))
        return {item[0]: item[1] for item in profile_zipped}
