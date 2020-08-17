import scrapy
from movie.Item.ImdbItem import ImdbItem

class ImdbSpider(scrapy.Spider):
    name = 'imdb'
    allowed_domains = ['www.imdb.com']
    start_urls = ['https://www.imdb.com/chart/top']

    custom_settings = {

        'DEFAULT_REQUEST_HEADERS' :{
            'Referer': 'https://www.imdb.com/chart/top',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25'
        }
    }

    def parse(self, response):

        movie_list = response.xpath('//tbody[@class="lister-list"]/tr')

        for i_item in movie_list:
            Imdb_Item = ImdbItem()

            Imdb_Item['movie_index'] = i_item.xpath('.//td[@class="titleColumn"]/text()').extract_first()

            Imdb_Item['movie_name'] = i_item.xpath('.//td[@class="titleColumn"]/a/text()').extract_first()

            Imdb_Item['movie_score'] = i_item.xpath('.//td[@class="ratingColumn imdbRating"]/strong/text()').extract_first()

            yield Imdb_Item
          
   

