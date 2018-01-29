import datetime
import os
import scrapy
import sqlite3
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
        # 'DEPTH_PRIORITY': 0,
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
        '主な勝鞍': "highlight_race", '近親馬': 'relatives',
        '募集情報': 'offer_info'
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

    START_DATE = '2000-01-08'

    def __init__(self, *args, **kwargs):
        # Get faculty link for each university
        super(NetKeibaCrawler, self).__init__(*args, **kwargs)
        self.allowed_domains = list(NetKeibaCrawler.DOMAIN_URL)

        # Create a database to store links produced during crawling, with status specifying the link is parsed
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        parent_path = os.path.join(parent_path, '..')
        database_path = os.path.join(parent_path, 'data/history.db')
        self.connection = sqlite3.connect(database_path)
        self.cursor = self.connection.cursor()

        error_db_path = os.path.join(parent_path, 'data/error.db')
        self.connection_exp = sqlite3.connect(error_db_path)
        self.cursor_exp = self.connection_exp.cursor()

        # SQL CREATE statement
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS crawl_history (link TEXT PRIMARY KEY, 
                                                                         parsed INTEGER, 
                                                                         parse_level TEXT,
                                                                         meta_data TEXT);''')
        self.connection.commit()
        self.cursor_exp.execute('''CREATE TABLE IF NOT EXISTS error_history (error_msg TEXT, error_time TEXT)''')
        self.connection_exp.commit()

        self.cursor.execute('SELECT * FROM crawl_history WHERE parsed = 0')
        self.record_exist = bool(self.cursor.fetchone())

        # Create link file to filter browsed link
        try:
            self.links_read = open(os.path.join(parent_path, 'data/links.txt'), 'r')
        except FileNotFoundError:
            # If file does not exist
            new_file = open(os.path.join(parent_path, 'data/links.txt'), 'w')
            new_file.close()
            self.links_read = open(os.path.join(parent_path, 'data/links.txt'), 'r')

        self.links = self.links_read.read().splitlines()
        # Convert to dictionary for faster processing
        self.links = {link: True for link in self.links}
        self.links_read.close()
        self.links_append_path = os.path.join(parent_path, 'data/links.txt')

        # National events are only held on weekend and hence filter out irrelevant date
        if not self.record_exist:
            start_date = datetime.datetime.strptime(NetKeibaCrawler.START_DATE, '%Y-%m-%d').date()
            self.dates = []
            while start_date <= datetime.datetime.utcnow().date():
                if (start_date.weekday() == 5) | (start_date.weekday() == 6):
                    self.dates.append(str(start_date))
                start_date += datetime.timedelta(days=1)
            for date in self.dates:
                link = 'http://db.netkeiba.com/race/list/%s/' % date.replace('-', '')
                # SQL INSERT statement, separated from yield statement
                self.cursor.execute('''INSERT INTO crawl_history (link, parsed, parse_level, meta_data)
                                                       values (?, ?, ?, ?)''',
                                    (link, 0, 'Race List', str({'url_requested': link, 'date': date})))
                self.connection.commit()

    def start_requests(self):
        # Add indexing for debugging
        # for date in self.dates[:1]:
        if not self.record_exist:
            self.logger.info('No history found and start from date %s' % NetKeibaCrawler.START_DATE)
            for date in self.dates:
                # Yielding request and provide relevant meta data
                link = 'http://db.netkeiba.com/race/list/%s/' % date.replace('-', '')
                request = Request(link, callback=self.parse, errback=self.errback_handling)
                request.meta['date'] = date
                request.meta['url_requested'] = link
                yield request
        else:
            self.logger.info('Found crawl history and start crawling from the given link list')
            # Jump to the corresponding level of parsing
            link_list = self.cursor.execute('SELECT * FROM crawl_history WHERE parsed = 0').fetchall()
            for record in link_list:
                link_request = record[0]
                if self.is_duplicate(link_request):
                    continue
                parse_level = record[2]
                meta_data = eval(record[3])
                callback_method = None

                # Get the corresponding callback
                if parse_level == 'Race List':
                    callback_method = self.parse
                elif parse_level == 'Race Record':
                    callback_method = self.parse_race
                elif parse_level == 'Horse Record':
                    callback_method = self.parse_horse
                elif parse_level == 'Jockey Record':
                    callback_method = self.parse_jockey
                elif parse_level == 'Owner Record':
                    callback_method = self.parse_owner
                elif parse_level == 'Trainer Record':
                    callback_method = self.parse_trainer
                elif parse_level == 'Breeder Record':
                    callback_method = self.parse_breeder
                elif parse_level == 'Parent Intermediate Step':
                    callback_method = self.parse_horse_breed
                elif parse_level == 'Jockey Profile':
                    callback_method = self.parse_jockey_profile
                elif parse_level == 'Trainer Profile':
                    callback_method = self.parse_trainer_profile
                if callback_method is not None:
                    yield Request(link_request, callback=callback_method, meta=meta_data, errback=self.errback_handling)

    def parse(self, response):
        # Parse page content at the top level
        self.logger.info('Parsing at race list %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

        # Define XPath to extract information from the page
        top_path = '//dl[@class[contains(., "race_top_hold_list")]]'
        sub_path_list = list(map(lambda path: top_path + '/%s' % path, [
            'dt/p/text()',  # Place
            'dd//dt/text()',  # Race Number
            '/dd//a/@title',  # Race Title
            '/dd//a/@href[not(contains(., "movie"))]'  # Href
        ]))
        place_list = list(map(lambda name: name.strip(), response.xpath(sub_path_list[0]).extract()))
        race_num_list = response.xpath(sub_path_list[1]).extract()
        title_list = response.xpath(sub_path_list[2]).extract()
        href_list = response.xpath(sub_path_list[3]).extract()

        # Re-organize information order and store into a temporary dictionary for next-level parsing
        race_list = []
        for race_num, title, link in zip(race_num_list, title_list, href_list):
            try:
                race_list.append([int(race_num.strip('R')), title, link])
            except ValueError:
                pass
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
            link_request = response.urljoin(value)
            if self.is_duplicate(link_request):
                continue
            meta_list = key.split(' ')
            target_meta = {
                'date': response.meta['date'],
                'custom': {
                    'date': meta_list[0], 'place': meta_list[1], 'race': meta_list[2], 'title': meta_list[3]
                },
                'url_requested': link_request
            }
            # SQL INSERT statement, separated from yield statement
            self.cursor.execute('INSERT INTO crawl_history (link, parsed, parse_level, meta_data) values (?, ?, ?, ?)',
                                (link_request, 0, 'Race Record', str(target_meta)))
            self.connection.commit()

        for key, value in link_dict.items():
            # Initiate new meta data
            link_request = response.urljoin(value)
            meta_list = key.split(' ')
            if self.is_duplicate(link_request):
                # self.logger.info('Found and filtered duplicate %s' % link_request)
                continue
            target_meta = {
                'date': response.meta['date'],
                'custom': {
                    'date': meta_list[0], 'place': meta_list[1], 'race': meta_list[2], 'title': meta_list[3]
                },
                'url_requested': link_request
            }
            yield response.follow(value, meta=target_meta, callback=self.parse_race, errback=self.errback_handling)

    def parse_race(self, response):
        # Get basic information of the current race record page
        self.logger.info('Parsing race %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

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
                row_element = [element for element in row_element if element != '**']
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
                link_request = response.urljoin(link)
                if self.is_duplicate(link_request):
                    continue
                curr_record.update({'url_requested': link_request})
                if 'horse' in link:
                    # SQL INSERT statement, separated from yield statement
                    self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                           values (?, ?, ?, ?)''', (link_request, 0, 'Horse Record', str(curr_record)))
                    self.connection.commit()
                elif 'jockey' in link:
                    # SQL INSERT statement, separated from yield statement
                    self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                           values (?, ?, ?, ?)''', (link_request, 0, 'Jockey Record',
                                                                    str(curr_record)))
                    self.connection.commit()
                elif 'owner' in link:
                    # SQL INSERT statement, separated from yield statement
                    self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                           values (?, ?, ?, ?)''', (link_request, 0, 'Owner Record',
                                                                    str(curr_record)))
                    self.connection.commit()
                elif 'trainer' in link:
                    # SQL INSERT statement, separated from yield statement
                    self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                           values (?, ?, ?, ?)''', (link_request, 0, 'Trainer Record',
                                                                    str(curr_record)))
                    self.connection.commit()

            for link in sorted(link_element):
                link_request = response.urljoin(link)
                if self.is_duplicate(link_request):
                    # self.logger.info('Found and filtered duplicate %s' % link_request)
                    continue
                curr_record.update({'url_requested': link_request})
                if 'horse' in link:
                    yield response.follow(link, callback=self.parse_horse, meta=curr_record,
                                          errback=self.errback_handling)
                elif 'jockey' in link:
                    yield response.follow(self.format_link(link, 'result'),
                                          callback=self.parse_jockey, meta=curr_record, errback=self.errback_handling)
                elif 'owner' in link:
                    yield response.follow(self.format_link(link, 'result'), callback=self.parse_owner, meta=curr_record,
                                          errback=self.errback_handling)
                elif 'trainer' in link:
                    yield response.follow(self.format_link(link, 'result'),
                                          callback=self.parse_trainer, meta=curr_record, errback=self.errback_handling)

    def parse_horse_breed(self, response):
        # Intermediary step for parent horse information crawling
        self.logger.info('Parsing parent %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()
        content = list(map(lambda text: html.fromstring(text), response.xpath('//td[@rowspan="16"]').extract()))
        link_list = [None if len(element.xpath('a/@href')) <= 0 else element.xpath('a/@href')[0] for element in content]
        for link in link_list:
            if self.is_duplicate(response.urljoin(link)):
                continue
            new_meta = response.meta.copy()
            new_meta['url_requested'] = response.urljoin(link)
            # SQL INSERT statement, separated from yield statement
            self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                   values (?, ?, ?, ?)''',
                                (response.urljoin(link), 0, 'Horse Record', str(new_meta)))
            self.connection.commit()

        for link in link_list:
            if self.is_duplicate(response.urljoin(link)):
                # self.logger.info('Found and filtered duplicate %s' % response.urljoin(link))
                continue
            new_meta = response.meta.copy()
            new_meta['url_requested'] = response.urljoin(link)
            yield response.follow(link, callback=self.parse_horse, meta=new_meta, errback=self.errback_handling)

    def parse_horse(self, response):
        self.logger.info('Parsing horse %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

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

        # Extract parent links
        parent = list(map(lambda text: html.fromstring(text), response.xpath('//td[@rowspan="2"]').extract()))
        parent = {' '.join(element.xpath('//text()[normalize-space(.)]')): '-' if len(element.xpath('//a/@href')) <= 0
                  else element.xpath('//a/@href')[0] for element in parent}
        profile_dict.update({'parents': ' '.join(list(parent.keys()))})

        # Yield item of horse record
        horse_record = HorseRecord(profile_dict)
        yield horse_record

        # Extract breeder information
        breeder_link = response.xpath('//a[@href[contains(., "breeder/")]]/@href').extract_first()
        breeder_meta = response.meta.copy()
        breeder_meta['breeder_name'] = profile_dict[u'breeder']
        if breeder_link is not None:
            breeder_link_request = self.format_link(breeder_link, 'result')
            if self.is_duplicate(response.urljoin(breeder_link_request)):
                pass
            else:
                breeder_meta.update({'url_requested': response.urljoin(breeder_link_request)})
                # SQL INSERT statement
                self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                       values (?, ?, ?, ?)''',
                                    (response.urljoin(breeder_link_request), 0, 'Breeder Record', str(breeder_meta)))
                self.connection.commit()

        if breeder_link is not None:
            breeder_link_request = self.format_link(breeder_link, 'result')
            if self.is_duplicate(response.urljoin(breeder_link_request)):
                # self.logger.info('Found and filtered duplicate %s' % response.urljoin(breeder_link_request))
                pass
            else:
                yield response.follow(breeder_link_request,
                                      callback=self.parse_breeder, meta=breeder_meta, errback=self.errback_handling)

        # Get parent information
        if not response.meta.get('parent', False):
            for key, value in parent.items():
                link_request = response.urljoin(value)
                if self.is_duplicate(link_request):
                    continue
                new_meta = response.meta.copy()
                new_meta['depth'] = response.meta['depth'] - 1
                new_meta['parent'] = True
                new_meta['url_requested'] = link_request
                # SQL INSERT statement, separated from yield statement
                self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                       values (?, ?, ?, ?)''',
                                    (link_request, 0, 'Parent Intermediate Step', str(new_meta)))
                self.connection.commit()

            for key, value in parent.items():
                link_request = response.urljoin(value)
                if self.is_duplicate(link_request):
                    # self.logger.info('Found and filtered duplicate %s' % link_request)
                    continue
                response.meta['depth'] -= 1
                new_meta = response.meta.copy()
                new_meta['depth'] = response.meta['depth']
                new_meta['parent'] = True
                new_meta['url_requested'] = link_request
                yield response.follow(value, callback=self.parse_horse_breed, meta=new_meta,
                                      errback=self.errback_handling)

    def parse_breeder(self, response):
        self.logger.info('Parsing breeder %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

        # Get table content and basic information
        row_data = self.get_table_rows(response)[1][2:]
        breeder_name = response.meta['breeder_name']
        for row_element in row_data:
            # Yield item of breeder record
            breeder_record = dict(zip(NetKeibaCrawler.INDIVIDUAL_COLUMNS, [u'生産者', breeder_name] + row_element))
            breeder_record = IndividualRecord(breeder_record)
            yield breeder_record

    def parse_owner(self, response):
        self.logger.info('Parsing owner %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

        # Get table content and basic information
        row_data = self.get_table_rows(response)[1][2:]
        owner_name = response.meta['record'][-2]
        for row_element in row_data:
            # Yield item of owner record
            owner_record = dict(zip(NetKeibaCrawler.INDIVIDUAL_COLUMNS, [u'馬主', owner_name] + row_element))
            owner_record = IndividualRecord(owner_record)
            yield owner_record

    def parse_jockey(self, response):
        self.logger.info('Parsing jockey %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

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
            'url_requested': profile_link
        }

        if self.is_duplicate(profile_link):
            # self.logger.info('Found and filtered duplicate %s' % profile_link)
            pass
        else:
            # SQL INSERT statement
            self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                   values (?, ?, ?, ?)''',
                                (profile_link, 0, 'Jockey Profile', str(new_meta)))
            self.connection.commit()
            yield Request(profile_link, callback=self.parse_jockey_profile, meta=new_meta,
                          errback=self.errback_handling)

    def parse_trainer(self, response):
        self.logger.info('Parsing trainer %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

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
            'url_requested': profile_link
        }

        if self.is_duplicate(profile_link):
            # self.logger.info('Found and filtered duplicate %s' % profile_link)
            pass
        else:
            # SQL INSERT statement
            self.cursor.execute('''INSERT OR IGNORE INTO crawl_history (link, parsed, parse_level, meta_data) 
                                   values (?, ?, ?, ?)''',
                                (profile_link, 0, 'Trainer Profile', str(new_meta)))
            self.connection.commit()
            yield Request(profile_link, callback=self.parse_trainer_profile, meta=new_meta,
                          errback=self.errback_handling)

    def parse_jockey_profile(self, response):
        self.logger.info('Parsing jockey profile %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

        # Get profile information
        profile_info = self.get_profile_table(response)
        profile_info = {key if key not in NetKeibaCrawler.JOCKEY_COLUMNS.keys() else
                        NetKeibaCrawler.JOCKEY_COLUMNS[key]: value for key, value in profile_info.items()}
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
        self.logger.info('Parsing trainer profile %s' % response.url)
        self.links[response.meta['url_requested']] = True
        with open(self.links_append_path, 'a') as f:
            f.write(response.meta['url_requested'] + '\n')
            f.close()

        # SQL DELETE statement
        self.cursor.execute('DELETE FROM crawl_history WHERE link = ?', (response.meta['url_requested'], ))
        self.connection.commit()

        # Get profile information
        profile_info = self.get_profile_table(response)
        profile_info = {key if key not in NetKeibaCrawler.TRAINER_COLUMNS.keys()
                        else NetKeibaCrawler.TRAINER_COLUMNS[key]: value for key, value in profile_info.items()}
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

    def errback_handling(self, failure):
        # Handle unexpected error
        self.cursor_exp.execute('''INSERT OR IGNORE INTO error_history (error_msg, error_time) 
                                   values (?, ?)''', (repr(failure), str(datetime.datetime.now())))
        self.connection_exp.commit()

    def is_duplicate(self, link_address):
        # To tell whether there is a duplicate of link requested
        try:
            if self.links[link_address]:
                return True
        except KeyError:
            return False

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
        return {item[0]: '-' if len(item) <= 1 else item[1] for item in profile_zipped}

    @staticmethod
    def horse_record_translate(original_string):
        try:
            return NetKeibaCrawler.HORSE_COLUMNS[original_string]
        except KeyError:
            return original_string
