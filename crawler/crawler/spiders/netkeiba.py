import datetime
import os
import scrapy
from lxml import html
from scrapy import Request
from crawler.crawler.items import RaceRecord, HorseRecord, IndividualRecord, TrainerProfile, JockeyProfile


class NetKeibaCrawler(scrapy.Spider):

    # This class targets at crawling race records from Netkeiba by first storing race link and subsequently
    # yield race record items as defined in items.py

    name = 'netkeiba'
    custom_settings = {
        # Override custom settings preset by scrapy
        # Take a depth-first search algorithm by setting a negative priority or positive otherwise
        'DEPTH_LIMIT': 4,
        'DEPTH_PRIORITY': -4,
        'DEPTH_STATS_VERBOSE': True,

        # Limit the concurrent request per domain and moderate the server load
        'CONCURRENT_REQUESTS': 16,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'DOWNLOAD_DELAY': 0.5,
    }

    DOMAIN_URL = ['db.netkeiba.com']
    RACE_COLUMNS = [
        'run_date', 'place', 'race', 'title', 'type', 'track', 'distance', 'weather', 'condition', 'time',
        'finishing_position', 'bracket', 'horse_number', 'horse', 'sex_age', 'jockey_weight', 'jockey',
        'run_time', 'margin', 'corner_position', 'run_time_last_600', 'win_odds', 'win_fav', 'horse_weight',
        'trainer', 'breeder', 'prize'
    ]
    HORSE_COLUMNS = {
        '生年月日': 'date_of_birth',
        '調教師': 'trainer', '馬主': 'owner', '生産者': 'breeder',
        '産地': 'place_of_birth', 'セリ取引価格': 'transaction_price',
        '獲得賞金': 'prize_obtained', '通算成績': 'race_record',
        '主な勝鞍': "highlight_race", '近親馬': 'relatives'
    }
    INDIVIDUAL_COLUMNS = [
        'individual_type', 'name', 'year', 'rank', 'first', 'second', 'third', 'out',
        'races_major', 'wins_major', 'races_special', 'wins_special',
        'races_flat', 'wins_flat', 'races_grass', 'wins_grass', 'races_dirt', 'wins_dirt',
        'wins_percent', 'wins_percent_2nd', 'wins_percent_3rd', 'prize_obtained', 'representative_horse'
    ]
    TRAINER_COLUMNS = {
        '出身地': 'place_of_birth',
        '初出走日': 'first_run_date', '初出走馬': 'first_run_horse',
        '初勝利日': 'first_win_date', '初勝利馬': 'first_win_horse'
    }
    JOCKEY_COLUMNS = {
        '出身地': 'place_of_birth', '血液型': 'blood_type', '身長': 'height', '体重': 'weight',
        '平地初騎乗日': 'first_flat_run_date', '平地初騎乗馬': 'first_flat_run_horse',
        '平地初勝利日': 'first_flat_win_date', '平地初勝利馬': 'first_flat_win_horse',
        '障害初騎乗日': 'first_obs_run_date', '障害初騎乗馬': 'first_obs_run_horse',
        '障害初勝利日': 'first_obs_win_date', '障害初勝利馬': 'first_obs_win_horse'
    }

    # Start from the earliest available date
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) + '/status.log', 'r') as f:
        START_DATE = f.readline()
        START_DATE = START_DATE if START_DATE != '' else '2000-01-06'
        f.close()

    def __init__(self, *args, **kwargs):
        # Get faculty link for each university
        super(NetKeibaCrawler, self).__init__(*args, **kwargs)
        self.allowed_domains = list(NetKeibaCrawler.DOMAIN_URL)
        start_date = datetime.datetime.strptime(NetKeibaCrawler.START_DATE, '%Y-%m-%d').date()
        self.logger.info('Start crawling from date %s' % start_date)
        self.dates = []

        # National events are only held on weekend and hence filter out irrelevant date
        while start_date <= datetime.datetime.utcnow().date():
            if (start_date.weekday() == 5) | (start_date.weekday() == 6):
                self.dates.append(str(start_date))
            start_date += datetime.timedelta(days=1)

    def start_requests(self):
        # Add indexing for debugging
        # for date in self.dates[:1]:
        for date in self.dates:
            # Yielding request and provide relevant meta data
            request = Request('http://db.netkeiba.com/race/list/%s/' % date.replace('-', ''), callback=self.parse)
            request.meta['date'] = date
            yield request

    def parse(self, response):
        # Parse page content at the top level
        self.logger.info('Parsing at race list %s' % response.url)

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
        # for element in race_list[:1]:
        for element in race_list:
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
                'date': response.meta['date'],
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
        self.logger.info('Parsing race %s' % response.url)
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) + '/status.log', 'w') as f:
            f.write(response.meta['date'])
            f.close()

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
                # Example:
                # ['2000-01-08', '中山', '1R', '4歳未勝利', 'ダ', '右', '1200m', '曇', '良', '10:00',
                # '1', '6', '8', 'コスモイーグル', '牡4', '54', '村田一誠', '1:13.9', '-', '4-3', '38.9', '1.7',
                # '1', '458(-2)', '[東]稲葉隆一', '岡田美佐子', '510.0']
                'record': list(response.meta['custom'].values()) + basic_info + row_element
            }

            # Yield item of race record
            race_record = RaceRecord(dict(zip(NetKeibaCrawler.RACE_COLUMNS, curr_record['record'])))
            yield race_record

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
            'horse_name': response.meta['record'][13],
            'status': basic_info[0],
            'gender': basic_info[1],
            'breed': basic_info[2]
        }

        # Extract profile of the current horse page
        profile = response.xpath('//table[@class[contains(., "db_prof_table")]]//tr').extract()
        profile_read = list(map(lambda text: html.fromstring(text), profile))
        profile_zipped = list(map(lambda content: content.xpath('//text()[normalize-space(.)]'), profile_read))
        profile_zipped = list(map(lambda x: [x[0], ' '.join(x[1:]).strip()], profile_zipped))
        profile_dict = {self.horse_record_translate(item[0]): item[1] for item in profile_zipped}
        profile_dict.update(info_dict)

        # Yield item of horse record
        horse_record = HorseRecord(profile_dict)
        yield horse_record

        # Extract breeder information
        breeder_link = response.xpath('//a[@href[contains(., "breeder/")]]/@href').extract_first()
        breeder_meta = response.meta.copy()
        breeder_meta['breeder_name'] = profile_dict[u'breeder']
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
                response.meta['depth'] -= 1
                new_meta = response.meta.copy()
                new_meta['depth'] = response.meta['depth']
                new_meta['parent'] = True
                yield response.follow(value, callback=self.parse_horse_breed, meta=new_meta)

    def parse_breeder(self, response):
        # Get table content and basic information
        row_data = self.get_table_rows(response)[1][2:]
        breeder_name = response.meta['breeder_name']
        for row_element in row_data:
            # Yield item of breeder record
            breeder_record = dict(zip(NetKeibaCrawler.INDIVIDUAL_COLUMNS, [u'生産者', breeder_name] + row_element))
            breeder_record = IndividualRecord(breeder_record)
            yield breeder_record

    def parse_owner(self, response):
        # Get table content and basic information
        row_data = self.get_table_rows(response)[1][2:]
        owner_name = response.meta['record'][-2]
        for row_element in row_data:
            # Yield item of owner record
            owner_record = dict(zip(NetKeibaCrawler.INDIVIDUAL_COLUMNS, [u'馬主', owner_name] + row_element))
            owner_record = IndividualRecord(owner_record)
            yield owner_record

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
            'trainer_name': trainer_name,
            'basic_info': basic_info,
        }
        yield Request(profile_link, callback=self.parse_trainer_profile, meta=new_meta)

    def parse_jockey_profile(self, response):
        # Get profile information
        profile_info = self.get_profile_table(response)
        profile_info = {NetKeibaCrawler.JOCKEY_COLUMNS[key]: value for key, value in profile_info.items()}
        profile_info.update(
            {
                'jockey_name': response.meta['jockey_name'],
                'date_of_birth': response.meta['basic_info']
            }
        )

        # Yield item of jockey record and profile
        jockey_profile = JockeyProfile(profile_info)
        yield jockey_profile
        for row_element in response.meta['row_data']:
            jockey_record = dict(zip(NetKeibaCrawler.INDIVIDUAL_COLUMNS,
                                     [u'騎手', response.meta['jockey_name']] + row_element))
            jockey_record = IndividualRecord(jockey_record)
            yield jockey_record

    def parse_trainer_profile(self, response):
        # Get profile information
        profile_info = self.get_profile_table(response)
        profile_info = {NetKeibaCrawler.TRAINER_COLUMNS[key]: value for key, value in profile_info.items()}
        profile_info.update(
            {
                'trainer_name': response.meta['trainer_name'],
                'date_of_birth': response.meta['basic_info']
            }
        )

        # Yield item of trainer record and profile
        trainer_profile = TrainerProfile(profile_info)
        yield trainer_profile
        for row_element in response.meta['row_data']:
            trainer_record = dict(zip(NetKeibaCrawler.INDIVIDUAL_COLUMNS,
                                      [u'調教師', response.meta['trainer_name']] + row_element))
            trainer_record = IndividualRecord(trainer_record)
            yield trainer_record

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

    @staticmethod
    def horse_record_translate(original_string):
        return NetKeibaCrawler.HORSE_COLUMNS[original_string]
