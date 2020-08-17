import scrapy
from movie.Item.ImdbItem import ImdbItem

class MaoyanSpider(scrapy.Spider):
    name = 'maoyan'
    #allowed_domains = ['maoyan.com']
    # start_urls = ['http://maoyan.com/']
    custom_settings = {

        'DEFAULT_REQUEST_HEADERS' :{
            'Referer':'https://maoyan.com/board/4?offset={}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25'
        }
    }
    def start_requests(self):
        
        for i in range(0,100,10):
            url = 'https://maoyan.com/board/4?offset={}'.format(i)
            yield scrapy.Request(url=url,callback= self.parse)
            #time.sleep(1)
            
    def parse(self, response):
        movie_list = response.xpath('//div[@class="main"]/dl/dd')

        for i_item in movie_list:
            MyItem = ImdbItem()

            MyItem['movie_index'] = i_item.xpath('./i/text()').extract_first()

            MyItem['movie_name']  = i_item.xpath('.//p[@class="name"]/a/text()').extract_first()

            score_integer = i_item.xpath('.//p[@class="score"]/i[@class="integer"]/text()').extract_first()

            score_fraction = i_item.xpath('.//p[@class="score"]/i[@class="fraction"]/text()').extract_first()

            MyItem['movie_score'] = score_integer + score_fraction

            yield MyItem
