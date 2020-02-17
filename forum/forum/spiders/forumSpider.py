import scrapy
import re
import os
import logging
from scrapy.utils.log import configure_logging

class forumSpider(scrapy.Spider):
    name = "forum"

    custom_settings = {
        # I use mobile user-agent because some forums have different html in the desktop versions but the same mobile one
        'USER_AGENT': 'Mozilla/5.0 (iPad; U; CPU OS 3_2_1 like Mac OS X; en-us) AppleWebKit/531.21.10 (KHTML, like Gecko) Mobile/7B405',
        # let scrapy change the number of pipelines and delays basing on reaction of the website
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_DEBUG': True,
        'DOWNLOAD_TIMEOUT': 15
    }
    # logging inside the file
    configure_logging(install_root_handler=False)
    logging.basicConfig(
        filename='forum/log.txt',
        format='%(levelname)s: %(message)s',
        level=logging.INFO
    )

    # in this method I automatically create the seeds for the crawler, the seeds are the pages on the web-site forumfree.it, each page contains 100 links to different forums
    def start_requests(self):
        # parse pages of forumfree.it
        urls_forumfree_pages = []
        # n is the amount of page (on forumfree) we would like to crawl
        # each page has 100 links to the forums
        # n must be the number in the range from 1 to 1157
        n = 4
        for i in range(3, n):
            urls_forumfree_pages.append('https://www.forumfree.it/?act=topforum&p=' + str(i))

        for url in urls_forumfree_pages:
            yield scrapy.Request(url=url, callback=self.parse_forumfree)
        '''
        # parse a forum
        forum_href = "https://orologi.forumfree.it"
        url = forum_href + "/?act=Members"
        yield scrapy.Request(url=url, callback=self.parse_forum_members_list, meta={"urls_users_pages": []})
        '''
        '''
        # parse a member of a forum
        #forum_href = "https://pulcinella291.forumfree.it"
        urls = ["https://orologi.forumfree.it/m/?act=Profile&MID=11391919",
                "https://orologi.forumfree.it/m/?act=Profile&MID=71364",
                "https://orologi.forumfree.it/m/?act=Profile&MID=67370",
                "https://orologi.forumfree.it/m/?act=Profile&MID=63010",
                "https://orologi.forumfree.it/m/?act=Profile&MID=62121",
                "https://orologi.forumfree.it/m/?act=Profile&MID=53281"]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_user_page)
        '''
        '''
        # parse a page with messeges (we need meta data about the user (like age, gender, location, usernema))
        forum_href = "https://orologi.forumfree.it"
        url = forum_href + "/m/?act=Search&MID=8131741&st=2250"
        yield scrapy.Request(url=url, callback=self.parse_posts, meta={"user_info": "info"})
        '''

    # parsing of the website "forumfree.it" to extract links to forums which have used forumfree platform to build the forum
    def parse_forumfree(self, response):
        # run through the page forumfree.it to extract links to forums
        urls_forum_links = []
        for forum_link_anchor in response.css("#top-list li"):
            forum_href = (forum_link_anchor.css('a::attr(href)').extract_first()).encode("utf-8")
            # append not just the forum href, but the page with members
            urls_forum_links.append(forum_href + "/?act=Members")
            with (open("forum/summary_name.txt", "a")) as f:
                f.write(forum_link_anchor.css('a::text').extract_first().encode("utf-8") + "\n")
            with (open("forum/summary_link.txt", "a")) as f:
                f.write(forum_href + "\n")
        # start the crawl on urls extracted
        for url in urls_forum_links:
            # in metadata we append an empty list, in this list we will safe all the user's pages
            yield scrapy.Request(url=url, callback=self.parse_forum_members_list, meta={"urls_users_pages": []})

    # parsing of a page of the forum, where we can see all the members of the forum
    def parse_forum_members_list(self, response):
        # download the list of previously crawled links of members' pages
        urls_users_pages = response.meta["urls_users_pages"]
        # get links to users' pages on this page of members
        for user_anchor in response.css(".details"):
            user_href = user_anchor.css("a::attr(href)").extract_first()
            # add the link to the list
            urls_users_pages.append(response.urljoin(user_href))
        # extract the next page of members
        next_page = response.css("li.next a::attr(href)").extract_first()
        # if there is next page we continue to parse pages with members
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(url=next_page, callback=self.parse_forum_members_list, meta={"urls_users_pages": urls_users_pages})
        # otherwise we start parsing the pages with user's info (we get all the urls from the list)
        else:
            for url in urls_users_pages:
                yield scrapy.Request(url=url, callback=self.parse_user_page)

    # parsing of the user's page to extract username, gender, age, location and the link to the posts made by this user
    def parse_user_page(self, response):
        # saving html of the page to the variabile body
        body = response.body
        # dictionary with the info about the user
        info = {}
        # getting user's nickname
        info["user"] = response.css(".u_nick::text").extract_first()
        # getting the id of the user, which is used in the urls (in case we will want to control the crawl result using browser)
        info["link"] = re.search(r'&MID=[0-9]+', response.url).group(0)
        hasPosts = False
        # regex to find out if the user wrote any posts
        matchPosts = re.search(r'<b>Messages</b><br>([0-9]+,[0-9]+)', body)
        hasGender = False
        # regex to find out if the user has specified the gender field
        matchGender = re.search(r'<dt>Gender</dt><dd><span class="[a-z]+">([A-Za-z]+)</span></dd>', body)
        hasAge = False
        # regex to find out if the user has specified the age field
        matchAge = re.search(r'<dt>Date of birth</dt><dd>([0-9][0-9]? [A-Za-z]+ [0-9]{4})</dd>', body)
        hasLocation = False
        # regex to find out if the user has specified the location field
        matchLocation = re.search(r'<dt>Location</dt><dd>([^<>]+)</dd>', body)
        # if have specified lets save it in the dictionary
        if matchPosts is not None:
            hasPosts = True
            info["posts"] = matchPosts.group(1)
        if matchGender is not None:
            hasGender = True
            info["gender"] = matchGender.group(1)
        if matchAge is not None:
            hasAge = True
            # refactor of the date from the format 6 January 1987 to 1987-01-06
            # split day, month and year
            split_date = re.split(" ", matchAge.group(1))
            # if the day is written like 6 we change it to be 06
            if re.match(r'[0-9][0-9]', split_date[0]) is None:
                split_date[0] = "0" + split_date[0]
            # change the name of the month to its number
            if split_date[1] == "January":
                split_date[1] = "01"
            elif split_date[1] == "February":
                split_date[1] = "02"
            elif split_date[1] == "March":
                split_date[1] = "03"
            elif split_date[1] == "April":
                split_date[1] = "04"
            elif split_date[1] == "May":
                split_date[1] = "05"
            elif split_date[1] == "June":
                split_date[1] = "06"
            elif split_date[1] == "July":
                split_date[1] = "07"
            elif split_date[1] == "August":
                split_date[1] = "08"
            elif split_date[1] == "September":
                split_date[1] = "09"
            elif split_date[1] == "October":
                split_date[1] = "10"
            elif split_date[1] == "November":
                split_date[1] = "11"
            elif split_date[1] == "December":
                split_date[1] = "12"
            # put together the date: first year, then month, then day
            info["age"] = split_date[2] + "-" + split_date[1] + "-" + split_date[0]
        if matchLocation is not None:
            hasLocation = True
            info["location"] = matchLocation.group(1)
        # we create the variable corpus_description to store the info about which type of user's data we have
        # we will then use this corpus description to store in separate files the messages of the users with different info available
        corpus_description = ""
        if hasPosts == True and hasAge == True and hasLocation == True and hasGender == True:
            corpus_description = "_full_info"
        '''
        elif hasPosts == True and hasGender == True and hasAge == True:
            corpus_description = "_no_location"
        elif hasPosts == True and hasLocation == True and hasGender == True:
            corpus_description = "_no_age"
        elif hasPosts == True and hasAge == True and hasLocation == True:
            corpus_description = "_no_gender"
        elif hasPosts == True and hasGender == True:
            corpus_description = "_no_age_no_location"
        elif hasPosts == True and hasLocation == True:
            corpus_description = "_no_gender_no_age"
        elif hasPosts == True and hasAge == True:
            corpus_description = "_no_gender_no_location"
        '''
        # if corpus_description is still an empty string it means that we don't have any info of interest about this user, so we don't crawl his posts
        # otherwise we extract the link to the posts which were written by this user and start to parse them
        if corpus_description != "":
            info["corpus_description"] = corpus_description
            posts_page = response.css("#u_msg a::attr(href)").extract_first()
            if posts_page is not None:
                posts_page = response.urljoin(posts_page)
                yield scrapy.Request(url=posts_page, callback=self.parse_posts, meta={"user_info": info})

    # parsing of the pages with posts
    def parse_posts(self, response):
        # extract info about the user from meta data
        info = response.meta["user_info"]
        # create a folder with the name of the forum
        forum_name = re.split('\.', re.split('/', response.url)[2])[0]
        corpus_path = "forum/result/" + forum_name
        if not os.path.exists(corpus_path):
            os.makedirs(corpus_path)
        corpus_path += "/" + forum_name + info["corpus_description"] + ".tsv"
        with open(corpus_path, 'a') as f:
            for mes in response.css(".post"):
                message = ""
                # we don't need to delete the citations, because "::text" feature takes only the text, which the tag contains, and ignore text in other tags inside this tag (for example, in the case: <td><span>abc</span>def</td> - td::text will ignore text inside the span and will take only "def"). As the citations on the forum are in their own div-s inside <td> tag, we can not worry about them getting inside the corpus
                for m in mes.css("td::text").extract():
                    # delete unuseful space and tab characters in the beggining and the end of the string
                    # if the post is written in several paragraphs we append them one to another, so they are in the same line
                    message += re.sub(r'\s*(.+)\s*', r'\1', m) + " "
                # extract the date when the post was written ([:10] means that we take only YYYY-MM-DD without exact time)
                time = mes.css(".timeago::attr(title)").extract_first()[:10]
                # depending on what info about the user we have, we write in file different info
                if info["corpus_description"] == "_full_info":
                    data = (u'' + info["user"] + '\t' + info["link"] + '\t' + info["age"] + '\t' + info["gender"] + '\t' + info["location"] + '\t' + message + '\t' + time + '\n').encode('utf-8')
                    f.write(data)
                '''
                elif info["corpus_description"] == "_no_location":
                    data = (u'' + info["user"] + '\t' + info["link"] + '\t' + info["age"] + '\t' + info["gender"] + '\t' + message + '\t' +
                            time + '\n').encode('utf-8')
                    f.write(data)
                elif info["corpus_description"] == "_no_age":
                    data = (u'' + info["user"] + '\t' + info["link"] + '\t' + info["gender"] + '\t' + info["location"] + '\t' + message + '\t' + time + '\n').encode('utf-8')
                    f.write(data)
                elif info["corpus_description"] == "_no_gender":
                    data = (u'' + info["user"] + '\t' + info["link"] + '\t' + info["age"] + '\t' + info["location"] + '\t' + message + '\t' +
                            time + '\n').encode('utf-8')
                    f.write(data)
                elif info["corpus_description"] == "_no_age_no_location":
                    data = (u'' + info["user"] + '\t' + info["link"] + '\t' + info["gender"] + '\t' + message + '\t' + time + '\n').encode('utf-8')
                    f.write(data)
                elif info["corpus_description"] == "_no_gender_no_age":
                    data = (u'' + info["user"] + '\t' + info["link"] + '\t' + info["location"] + '\t' + message + '\t' + time + '\n').encode('utf-8')
                    f.write(data)
                elif info["corpus_description"] == "_no_gender_no_location":
                    data = (u'' + info["user"] + '\t' + info["link"] + '\t' + info["age"] + '\t' + message + '\t' + time + '\n').encode('utf-8')
                    f.write(data)
                '''
        # if we are on the first page of posts, we extract number of pages and add all their links to the crawl
        first_page = re.search(r'>1/([0-9]+)', response.body)
        if first_page is not None and re.search('&st=', response.url) is None:
            str_num_pages = first_page.group(1)
            int_num_pages = int(str_num_pages)
            for i in range(1, int_num_pages):
                next_page = response.url + "&st=" + str(15 * i)
                yield scrapy.Request(url=next_page, callback=self.parse_posts, meta={"user_info": info})
        # I have tried to use next_page mechanism of the pages with posts, but sometimes if the user wrote a lot of post (for example, 1000 pages with 15 posts) the crawler wouldn't find li.next on approximately 300-th page. As I didn't know why it was happening I have decided to create all the urls "by hand"
        '''
        next_page = response.css("li.next a::attr(href)").extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            #yield scrapy.Request(url=next_page, callback=self.parse_posts, meta={"user_info": info})
            with open("forum/results/user.txt", "w") as f:
                f.write(response.url + "\n\n" + re.search('1/([0-9]+)', response.body).group(1) + "\n\n" + response.body)
        '''
