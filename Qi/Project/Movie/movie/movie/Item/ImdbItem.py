import scrapy

class ImdbItem(scrapy.Item):

    movie_name  = scrapy.Field()

    movie_index = scrapy.Field()

    movie_score = scrapy.Field()

    