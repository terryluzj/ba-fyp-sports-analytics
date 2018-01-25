# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os
import sqlite3
from crawler.crawler.items import RaceRecord, HorseRecord, IndividualRecord, TrainerProfile, JockeyProfile


class CrawlerPipeline(object):

    def __init__(self):
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        database_path = os.path.join(parent_path, 'data/race.db')

        # Establish database connection
        self.connection = sqlite3.connect(database_path)
        self.cursor = self.connection.cursor()

        # Create race record table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS race_record
            (
                run_date DATETIME, place TEXT, race TEXT, title TEXT, type TEXT, track TEXT, distance TEXT,
                weather TEXT, condition TEXT, time TEXT,
                finishing_position TEXT, bracket TEXT, horse_number TEXT, horse TEXT, sex_age TEXT,
                jockey_weight TEXT, jockey TEXT, run_time TEXT, margin TEXT, corner_position TEXT,
                run_time_last_600 TEXT, win_odds TEXT, win_fav TEXT, horse_weight TEXT,
                trainer TEXT, breeder TEXT, prize TEXT,
                PRIMARY KEY (run_date, place, race, title, horse)
            );
        ''')

        # Create horse record table
        self.cursor.execute('''        
            CREATE TABLE IF NOT EXISTS horse_record
            (
                horse_name TEXT PRIMARY KEY,
                date_of_birth DATETIME, trainer TEXT, owner TEXT, breeder TEXT, place_of_birth TEXT, 
                transaction_price TEXT, prize_obtained TEXT, race_record TEXT, highlight_race TEXT,
                relatives TEXT, status TEXT, gender TEXT, breed TEXT, offer_info TEXT
            );
        ''')

        # Create individual record table
        self.cursor.execute('''        
            CREATE TABLE IF NOT EXISTS individual_record
            (
                individual_type TEXT, name TEXT , year TEXT, rank TEXT, first TEXT, second TEXT, third TEXT, out TEXT,
                races_major TEXT, wins_major TEXT, races_special TEXT, wins_special TEXT,
                races_flat TEXT, wins_flat TEXT, races_grass TEXT, wins_grass TEXT, races_dirt TEXT, wins_dirt TEXT,
                wins_percent TEXT, wins_percent_2nd TEXT, wins_percent_3rd TEXT,
                prize_obtained TEXT, representative_horse TEXT,
                PRIMARY KEY (name, year)
            );
        ''')

        # Create trainer profile table
        self.cursor.execute('''        
            CREATE TABLE IF NOT EXISTS trainer_profile
            (
                trainer_name TEXT PRIMARY KEY, date_of_birth TEXT, place_of_birth TEXT,
                first_run_date TEXT, first_run_horse TEXT, first_win_date TEXT, first_win_horse TEXT
            );
        ''')

        # Create jockey profile table
        self.cursor.execute('''        
            CREATE TABLE IF NOT EXISTS jockey_profile
            (
                jockey_name TEXT PRIMARY KEY, date_of_birth TEXT, place_of_birth TEXT, blood_type TEXT,
                height TEXT, weight TEXT,
                first_flat_run_date TEXT, first_flat_run_horse TEXT, 
                first_flat_win_date TEXT, first_flat_win_horse TEXT,
                first_obs_run_date TEXT, first_obs_run_horse TEXT,
                first_obs_win_date TEXT, first_obs_win_horse TEXT
            );
        ''')

    def process_item(self, item, spider):
        if isinstance(item, RaceRecord):
            self.cursor.execute('''
                INSERT OR IGNORE INTO race_record 
                (run_date, place, race, title, type, track, distance, weather, condition, time, 
                 finishing_position, bracket, horse_number, horse, sex_age, jockey_weight, jockey, run_time, margin,
                 corner_position, run_time_last_600, win_odds, win_fav, horse_weight, trainer, breeder, prize
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (item.get('run_date', 'null'), item.get('place', 'null'), item.get('race', 'null'),
                      item.get('title', 'null'), item.get('type', 'null'), item.get('track', 'null'),
                      item.get('distance', 'null'), item.get('weather', 'null'), item.get('condition', 'null'),
                      item.get('time', 'null'), item.get('finishing_position', 'null'), item.get('bracket', 'null'),
                      item.get('horse_number', 'null'), item.get('horse', 'null'), item.get('sex_age', 'null'),
                      item.get('jockey_weight', 'null'), item.get('jockey', 'null'), item.get('run_time', 'null'),
                      item.get('margin', 'null'), item.get('corner_position', 'null'),
                      item.get('run_time_last_600', 'null'), item.get('win_odds', 'null'),
                      item.get('win_fav', 'null'), item.get('horse_weight', 'null'), item.get('trainer', 'null'),
                      item.get('breeder', 'null'), item.get('prize', 'null'))
            )
        elif isinstance(item, HorseRecord):
            self.cursor.execute('''
                INSERT OR IGNORE INTO horse_record 
                (horse_name, date_of_birth, trainer, owner, breeder, place_of_birth, 
                 transaction_price, prize_obtained, race_record, highlight_race, relatives, status, gender, breed, 
                 offer_info
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (item.get('horse_name', 'null'), item.get('date_of_birth', 'null'), item.get('trainer', 'null'),
                      item.get('owner', 'null'), item.get('breeder', 'null'), item.get('place_of_birth', 'null'),
                      item.get('transaction_price', 'null'), item.get('prize_obtained', 'null'),
                      item.get('race_record', 'null'), item.get('highlight_race', 'null'),
                      item.get('relatives', 'null'), item.get('status', 'null'), item.get('gender', 'null'),
                      item.get('breed', 'null'), item.get('offer_info', 'null'))
            )
        elif isinstance(item, IndividualRecord):
            self.cursor.execute('''
                INSERT OR IGNORE INTO individual_record 
                (individual_type, name, year, rank, first, second, third, out, races_major, wins_major, races_special, 
                 wins_special, races_flat, wins_flat, races_grass, wins_grass, races_dirt, wins_dirt, wins_percent, 
                 wins_percent_2nd, wins_percent_3rd, prize_obtained, representative_horse
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (item.get('individual_type', 'null'), item.get('name', 'null'), item.get('year', 'null'),
                      item.get('rank', 'null'), item.get('first', 'null'), item.get('second', 'null'),
                      item.get('third', 'null'), item.get('out', 'null'), item.get('races_major', 'null'),
                      item.get('wins_major', 'null'), item.get('races_special', 'null'),
                      item.get('wins_special', 'null'), item.get('races_flat', 'null'), item.get('wins_flat', 'null'),
                      item.get('races_grass', 'null'), item.get('wins_grass', 'null'), item.get('races_dirt', 'null'),
                      item.get('wins_dirt', 'null'), item.get('wins_percent', 'null'),
                      item.get('wins_percent_2nd', 'null'), item.get('wins_percent_3rd', 'null'),
                      item.get('prize_obtained', 'null'), item.get('representative_horse', 'null'))
            )
        elif isinstance(item, TrainerProfile):
            self.cursor.execute('''
                INSERT OR IGNORE INTO trainer_profile
                (trainer_name, date_of_birth, place_of_birth, first_run_date, first_run_horse, first_win_date, 
                 first_win_horse
                ) values (?, ?, ?, ?, ?, ?, ?)
                ''', (item.get('trainer_name', 'null'), item.get('date_of_birth', 'null'),
                      item.get('place_of_birth', 'null'), item.get('first_run_date', 'null'),
                      item.get('first_run_horse', 'null'), item.get('first_win_date', 'null'),
                      item.get('first_win_horse', 'null'))
            )
        elif isinstance(item, JockeyProfile):
            self.cursor.execute('''
                INSERT OR IGNORE INTO jockey_profile
                (jockey_name, date_of_birth, place_of_birth, blood_type, height, weight, first_flat_run_date, 
                 first_flat_run_horse, first_flat_win_date, first_flat_win_horse, first_obs_run_date, 
                 first_obs_run_horse, first_obs_win_date, first_obs_win_horse
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (item.get('jockey_name', 'null'), item.get('date_of_birth', 'null'),
                      item.get('place_of_birth', 'null'), item.get('blood_type', 'null'),
                      item.get('height', 'null'), item.get('weight', 'null'), item.get('first_flat_run_date', 'null'),
                      item.get('first_flat_run_horse', 'null'), item.get('first_flat_win_date', 'null'),
                      item.get('first_flat_win_horse', 'null'), item.get('first_obs_run_date', 'null'),
                      item.get('first_obs_run_horse', 'null'), item.get('first_obs_win_date', 'null'),
                      item.get('first_obs_win_horse', 'null'))
            )

        self.connection.commit()
        return item
