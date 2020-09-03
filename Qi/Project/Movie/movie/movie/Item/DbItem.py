import scrapy

class DbItems(scrapy.Item):

    collection = u'DoubanMovie'

    # 排名
    movie_index = scrapy.Field()

    # 电影名称
    movie_name = scrapy.Field()

    #电影评分
    movie_score = scrapy.Field()

    #电影评论人数
    comment_num = scrapy.Field()
    