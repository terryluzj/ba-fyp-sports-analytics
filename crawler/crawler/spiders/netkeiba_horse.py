import datetime
import os
import scrapy
import sqlite3
from lxml import html
from scrapy import Request
from crawler.crawler.items import RaceRecord, HorseRecord, IndividualRecord, TrainerProfile, JockeyProfile


class NetKeibaHorseCrawler(scrapy.Spider):
	pass