# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field
from scrapy.loader.processors import TakeFirst


class RaceRecord(Item):
    # Basic Information
    run_date = Field(output_processor=TakeFirst())
    place = Field(output_processor=TakeFirst())
    race = Field(output_processor=TakeFirst())
    title = Field(output_processor=TakeFirst())
    type = Field(output_processor=TakeFirst())
    track = Field(output_processor=TakeFirst())
    distance = Field(output_processor=TakeFirst())
    weather = Field(output_processor=TakeFirst())
    condition = Field(output_processor=TakeFirst())
    time = Field(output_processor=TakeFirst())

    # Race-specific Information
    finishing_position = Field(output_processor=TakeFirst())
    bracket = Field(output_processor=TakeFirst())
    horse_number = Field(output_processor=TakeFirst())
    horse = Field(output_processor=TakeFirst())
    sex_age = Field(output_processor=TakeFirst())
    jockey_weight = Field(output_processor=TakeFirst())
    jockey = Field(output_processor=TakeFirst())
    run_time = Field(output_processor=TakeFirst())
    margin = Field(output_processor=TakeFirst())
    corner_position = Field(output_processor=TakeFirst())
    run_time_last_600 = Field(output_processor=TakeFirst())
    win_odds = Field(output_processor=TakeFirst())
    win_fav = Field(output_processor=TakeFirst())
    horse_weight = Field(output_processor=TakeFirst())
    trainer = Field(output_processor=TakeFirst())
    breeder = Field(output_processor=TakeFirst())
    prize = Field(output_processor=TakeFirst())

    def __repr__(self):
        return repr({'title': self['title']})


class HorseRecord(Item):
    # Detailed Information
    horse_name = Field(output_processor=TakeFirst())
    date_of_birth = Field(output_processor=TakeFirst())
    trainer = Field(output_processor=TakeFirst())
    owner = Field(output_processor=TakeFirst())
    breeder = Field(output_processor=TakeFirst())
    place_of_birth = Field(output_processor=TakeFirst())
    transaction_price = Field(output_processor=TakeFirst())
    prize_obtained = Field(output_processor=TakeFirst())
    race_record = Field(output_processor=TakeFirst())
    highlight_race = Field(output_processor=TakeFirst())
    relatives = Field(output_processor=TakeFirst())

    # Basic Information
    status = Field(output_processor=TakeFirst())
    gender = Field(output_processor=TakeFirst())
    breed = Field(output_processor=TakeFirst())

    def __repr__(self):
        return repr({'horse_name': self['horse_name']})


class IndividualRecord(Item):
    # Basic Information
    individual_type = Field(output_processor=TakeFirst())
    name = Field(output_processor=TakeFirst())
    year = Field(output_processor=TakeFirst())
    rank = Field(output_processor=TakeFirst())
    first = Field(output_processor=TakeFirst())
    second = Field(output_processor=TakeFirst())
    third = Field(output_processor=TakeFirst())
    out = Field(output_processor=TakeFirst())
    races_major = Field(output_processor=TakeFirst())
    wins_major = Field(output_processor=TakeFirst())
    races_special = Field(output_processor=TakeFirst())
    wins_special = Field(output_processor=TakeFirst())
    races_flat = Field(output_processor=TakeFirst())
    wins_flat = Field(output_processor=TakeFirst())
    races_grass = Field(output_processor=TakeFirst())
    wins_grass = Field(output_processor=TakeFirst())
    races_dirt = Field(output_processor=TakeFirst())
    wins_dirt = Field(output_processor=TakeFirst())
    wins_percent = Field(output_processor=TakeFirst())
    wins_percent_2nd = Field(output_processor=TakeFirst())
    wins_percent_3rd = Field(output_processor=TakeFirst())
    prize_obtained = Field(output_processor=TakeFirst())
    representative_horse = Field(output_processor=TakeFirst())

    def __repr__(self):
        return repr({'name': self['name'], 'year': self['year']})


class TrainerProfile(Item):
    trainer_name = Field(output_processor=TakeFirst())
    date_of_birth = Field(output_processor=TakeFirst())
    place_of_birth = Field(output_processor=TakeFirst())
    first_run_date = Field(output_processor=TakeFirst())
    first_run_horse = Field(output_processor=TakeFirst())
    first_win_date = Field(output_processor=TakeFirst())
    first_win_horse = Field(output_processor=TakeFirst())

    def __repr__(self):
        return repr({'trainer_name': self['trainer_name']})


class JockeyProfile(Item):
    jockey_name = Field(output_processor=TakeFirst())
    date_of_birth = Field(output_processor=TakeFirst())
    place_of_birth = Field(output_processor=TakeFirst())
    blood_type = Field(output_processor=TakeFirst())
    height = Field(output_processor=TakeFirst())
    weight = Field(output_processor=TakeFirst())
    first_flat_run_date = Field(output_processor=TakeFirst())
    first_flat_run_horse = Field(output_processor=TakeFirst())
    first_flat_win_date = Field(output_processor=TakeFirst())
    first_flat_win_horse = Field(output_processor=TakeFirst())
    first_obs_run_date = Field(output_processor=TakeFirst())
    first_obs_run_horse = Field(output_processor=TakeFirst())
    first_obs_win_date = Field(output_processor=TakeFirst())
    first_obs_win_horse = Field(output_processor=TakeFirst())

    def __repr__(self):
        return repr({'jockey_name': self['jockey_name']})
