import random
import asyncio
import time
from typing import List, Dict
from tqdm.auto import trange
from llm.localllm import LocalLLM
from graphragrec.prompts.prompts import QUERY
from graphragrec.utils.usage import calculateUsages
from graphragrec.schemas.query.reduce import ReduceTool


async def queryReduce(llm: LocalLLM,
                      query: str,
                      mapOutputs: List[Dict],
                      default_model: str = "gpt-4o"):
    message_content = f"Query: `{query}`\n MAP Outputs: ```{mapOutputs}```"
    messages = [{
        "role": "system",
        "content": QUERY.REDUCE
    }, {
        "role": "user",
        "content": message_content
    }]
    output, usage = await llm.__function_call__(
        messages,
        default_model,
        ReduceTool.tools,
        tool_choice=ReduceTool.tool_choice,
        max_tokens=4095)
    return output, usage


if __name__ == "__main__":
    import json
    import asyncio
    from configs import OPENAI_API_KEY
    user_query = "I want to have a fun weekend movie session with my bros."
    llm = LocalLLM(api_key=OPENAI_API_KEY)

    mapOutputs = [{
        "query_analysis":
        "The user query suggests a desire for an enjoyable and social experience with friends by watching movies together. This implies a preference for entertaining and engaging movies that are likely to foster a fun atmosphere, with genres such as comedy, action, or adventure being particularly suitable. The use of 'bros' indicates a movie session expected to appeal to a group of male friends, likely favoring movies with a more masculine appeal including action-packed films, comedies, and possibly some sports or crime dramas.",
        "query_decomposition": [{
            "query":
            "Fun weekend movie session",
            "explanation":
            "The user is looking for an enjoyable movie experience over a weekend, implying the movies should be entertaining, engaging, and possibly light-hearted."
        }, {
            "query":
            "With bros",
            "explanation":
            "Movies should appeal to a group of male friends, suggesting a bias towards action, comedy, or adventure genres. They might enjoy movies with camaraderie, and male bonds, and possibly films with a 'macho' edge."
        }],
        "final_relevant_communities": [{
            "community_id":
            13,
            "relevance_score":
            5,
            "relevance_reason":
            "This community focuses on movies with ensemble casts and intriguing plots, often involving action, relationships, and high-stakes scenarios which are well-suited for a fun, engaging session with friends.",
            "key_movies": [
                "Donnie Brasco", "Once Upon a Time in Hollywood",
                "Die Hard: With a Vengeance"
            ],
            "key_theme": ["action", "suspense", "relationships"]
        }, {
            "community_id":
            12,
            "relevance_score":
            4,
            "relevance_reason":
            "This community highlights notable movies with collaborative performances and interconnected talents. The diversity in themes and genres, along with acclaimed films, makes it a suitable choice for a varied yet enjoyable movie session.",
            "key_movies":
            ["Mystic River", "X-Men: First Class", "A Star is Born"],
            "key_theme": ["ensemble cast", "diverse genres", "relationships"]
        }, {
            "community_id":
            18,
            "relevance_score":
            4,
            "relevance_reason":
            "This community explores film industry connections with a focus on popular franchises and notable collaborations, which can provide a fun and nostalgic viewing experience for a group of friends.",
            "key_movies":
            ["Harry Potter series", "Home Alone", "Battle Royale"],
            "key_theme": ["franchises", "popular films", "ensemble cast"]
        }, {
            "community_id":
            16,
            "relevance_score":
            3,
            "relevance_reason":
            "This community revolves around high-profile actors and filmmakers from notable film series, offering a mix of epic fantasy and impactful drama, which can be entertaining.",
            "key_movies":
            ["The Lord of the Rings series", "Eastern Promises", "Green Book"],
            "key_theme":
            ["epic fantasy", "drama", "actor-director collaborations"]
        }],
        "irrelevant_communities": ["4", "9", "10", "11"]
    }, {
        "query_analysis":
        "The user's query suggests a desire for a lighthearted and enjoyable movie experience suitable for a group of friends \"bros\". The phrase \"fun weekend movie session\" implies a preference for movies that are entertaining, likely in genres such as comedy, action, or adventure. These movies should be engaging and well-suited for a social viewing environment where camaraderie is key.",
        "query_decomposition": [{
            "query":
            "Fun action movies",
            "explanation":
            "Movies with exciting action sequences can provide a thrilling experience that is enjoyable within a group setting."
        }, {
            "query":
            "Comedies for friends",
            "explanation":
            "Comedies are often crowd-pleasers and are ideal for a fun and relaxed group viewing."
        }, {
            "query":
            "Movies with male bonding themes",
            "explanation":
            "Movies that focus on friendship and male bonding can resonate well with a group of male friends."
        }],
        "final_relevant_communities": [{
            "community_id":
            0,
            "relevance_score":
            3,
            "relevance_reason":
            "While this community provides a rich tapestry of cinema history and evolution, it includes modern actors like Ryan Reynolds who is associated with popular fun movies like 'Deadpool.'",
            "key_movies": ["Deadpool"],
            "keythemes": ["Protagonist with a sense of humor, action-comedy"]
        }, {
            "community_id":
            3,
            "relevance_score":
            4,
            "relevance_reason":
            "This community highlights interconnected actors and movies across diverse genres including notable action and comedy films that could be a hit in a social setting.",
            "key_movies": ["The Departed", "Heat", "Fight Club"],
            "keythemes": ["Intense action, thrillers, camaraderie"]
        }, {
            "community_id":
            2,
            "relevance_score":
            3,
            "relevance_reason":
            "Focuses on iconic actors and directors known for their impactful works including crime and drama genres. Movies like 'Heat' and 'Goodfellas' are high-energy and engaging for group viewings.",
            "key_movies": ["Heat", "Goodfellas"],
            "keythemes": ["Crime, drama, intense action"]
        }, {
            "community_id":
            5,
            "relevance_score":
            4,
            "relevance_reason":
            "Includes diverse genres and blockbuster films from notable actors and directors, making it relevant for providing a mix of action-packed and engaging stories.",
            "key_movies": ["Black Hawk Down", "Saving Private Ryan"],
            "keythemes": ["War, survival, resilience, ensemble casts"]
        }, {
            "community_id":
            8,
            "relevance_score":
            2,
            "relevance_reason":
            "Covers a wide array of movies but many themes are more serious in nature. However, films with lighter tones and outstanding narratives like 'Jab We Met' provide some relevance.",
            "key_movies": ["Fargo", "Gone Girl", "Jab We Met"],
            "keythemes": [
                "Diverse genres, some comedic elements, strong character-driven stories"
            ]
        }, {
            "community_id":
            7,
            "relevance_score":
            1,
            "relevance_reason":
            "Predominantly focused on drama, which may not align well with the desired fun and lighthearted criteria for a bro movie night.",
            "key_movies": ["The Dark Knight", "Schindler's List"],
            "keythemes":
            ["Intense dramatic themes, complexity in storytelling"]
        }, {
            "community_id":
            1,
            "relevance_score":
            1,
            "relevance_reason":
            "'The Peanut Butter Falcon' predominantly revolves around emotional and personal growth themes that may not align with an entertaining session focused on fun.",
            "key_movies": ["The Peanut Butter Falcon"],
            "keythemes": ["Emotional journey, personal growth"]
        }],
        "irrelevant_communities": ["6"]
    }, {
        "query_analysis":
        "The user is looking for movies that are fun to watch with a group of male friends over the weekend. This implies a preference for genres that are entertaining, perhaps humorous, and engaging without being overly cerebral. The films should provide a shared experience that strengthens camaraderie and enjoyment.",
        "query_decomposition": [{
            "query":
            "fun movies",
            "explanation":
            "The central theme of the request is to select movies that are 'fun'. This is a broad genre that could include action, comedy, and adventure films."
        }, {
            "query":
            "weekend movie session",
            "explanation":
            "The movies should be suitable for a relaxed viewing environment, possibly involving casual conversation and a lighthearted atmosphere."
        }, {
            "query":
            "movies with bros",
            "explanation":
            "The movies should appeal to a group of male friends, potentially focusing on themes of friendship, adventure, or humor."
        }],
        "final_relevant_communities": [{
            "community_id":
            26,
            "relevance_score":
            5,
            "relevance_reason":
            "The community emphasizes ensemble cast dynamics, collaborative relationships, and genre diversity which can be highly engaging for a group of male friends. The mix of action, comedy, and historical dramas makes it suitable for a fun and diverse movie session.",
            "key_movies":
            ["The Royal Tenenbaums", "Ghostbusters", "Lucky Number Slevin"],
            "key_theme": [
                "Ensemble Cast Dynamics", "Collaborative relationships",
                "Genre Diversity"
            ]
        }, {
            "community_id":
            24,
            "relevance_score":
            4,
            "relevance_reason":
            "This community's diversity in action, drama, and heist films offers variety for a movie session with friends. Films involving high-stakes situations and a blend of different themes such as space exploration or urban settings can be appealing.",
            "key_movies": ["Drive", "Crash", "WALL-E"],
            "key_theme":
            ["Diverse Themes", "High-Stakes Situations", "Urban Exploration"]
        }, {
            "community_id":
            27,
            "relevance_score":
            4,
            "relevance_reason":
            "The community highlights notable directors and popular actors in significant roles which can provide quality entertainment. The variety of genres including thriller, drama, and animated features can cater to different tastes within a group.",
            "key_movies":
            ["Night on Earth", "The Birds", "Straight Outta Compton"],
            "key_theme":
            ["Notable Directors", "Popular Actors", "Collaborative Efforts"]
        }, {
            "community_id":
            25,
            "relevance_score":
            3,
            "relevance_reason":
            "While focused on classic movies, the community showcases themes of friendship, camaraderie, and iconic characters, which can be engaging for a group. However, the classic nature may not appeal to all modern viewers.",
            "key_movies": ["King Kong", "Rio Bravo", "A Night at the Opera"],
            "key_theme": [
                "Friendship and Loyalty", "Classic Hollywood",
                "Iconic Characters"
            ]
        }],
        "irrelevant_communities": ["28", "31", "32", "33"]
    }, {
        "query_analysis":
        "The user's query suggests they are looking for movies that are enjoyable and suitable for a weekend session with male friends. The focus is on having fun, implying genres that are entertaining, possibly action-packed, comedic, or adventure-oriented. The query also suggests a social and relaxed viewing experience.",
        "query_decomposition": [{
            "query":
            "Fun weekend movie session",
            "explanation":
            "This part of the query focuses on the context and atmosphere of the movie session, indicating that the films should be light-hearted, entertaining, and suitable for a social gathering."
        }, {
            "query":
            "Movies for bros",
            "explanation":
            "This sub-query emphasizes the audience (male friends), suggesting genres and themes that typically appeal to a male demographic, such as action, comedy, sports, and adventure."
        }],
        "final_relevant_communities": [{
            "community_id":
            15,
            "relevance_score":
            4,
            "relevance_reason":
            "The community explores a diverse range of genres, including action ('300'), sci-fi ('Watchmen'), and thematic movies ('Lethal Weapon'). It offers variety and excitement, aligning with the user's query for a fun weekend movie session.",
            "key_movies": ["300", "Watchmen", "Lethal Weapon"],
            "key_theme": ["Action", "Sci-Fi", "Adventure"]
        }, {
            "community_id":
            21,
            "relevance_score":
            3,
            "relevance_reason":
            "This community's focus on classic films and iconic performances may not align perfectly with the idea of a 'fun' session but provides valuable insights into timeless movies that can be enjoyable in a group setting.",
            "key_movies": [
                "The Treasure of the Sierra Madre", "Wild Strawberries",
                "The Seventh Seal"
            ],
            "key_themes": ["Drama", "Adventure", "Thriller"]
        }, {
            "community_id":
            22,
            "relevance_score":
            5,
            "relevance_reason":
            "The community features fun, engaging movies with interconnected relationships, like 'Guardians of the Galaxy' and 'Coco.' These movies are likely to be entertaining and fit well for a weekend session.",
            "key_movies": ["Guardians of the Galaxy", "Coco", "Toy Story"],
            "key_themes": ["Action", "Comedy", "Animated"]
        }, {
            "community_id":
            23,
            "relevance_score":
            4,
            "relevance_reason":
            "Featuring a mix of adventure ('The Goonies'), romantic comedy ('When Harry Met Sally'), and thrillers ('Zodiac'), this community offers a diverse set of engaging and entertaining movies.",
            "key_movies": ["The Goonies", "When Harry Met Sally", "Zodiac"],
            "key_themes": ["Adventure", "Rom-Com", "Thriller"]
        }, {
            "community_id":
            19,
            "relevance_score":
            3,
            "relevance_reason":
            "While the focus is on iconic movies and collaborations, films like 'Gran Torino' and 'Dirty Harry' offer action and character depth that might appeal to a group of male friends.",
            "key_movies": ["Gran Torino", "Dirty Harry", "Where Eagles Dare"],
            "key_themes": ["Action", "Crime", "Thriller"]
        }],
        "irrelevant_communities": ["17", "14", "20"]
    }, {
        "query_analysis":
        "The user wants to enjoy a fun movie session with friends specifically mentioned as 'bros', indicating a preference for engaging, possibly action-oriented, and communal viewing experiences that are enjoyable in a group setting. This implies a preference for genres like action, adventure, comedy, or sci-fi rather than serious drama or introspective films.",
        "query_decomposition": [{
            "query":
            "Fun weekend movie session",
            "explanation":
            "Focusing on the 'fun' aspect, likely looking for light-hearted, entertaining, and possibly fast-paced movies suitable for a social gathering."
        }, {
            "query":
            "Movies suitable for watching with bros",
            "explanation":
            "Emphasizing group dynamics which may include action, comedy, or adventure genres that are engaging and entertaining for a group of friends."
        }],
        "final_relevant_communities": [{
            "community_id":
            29,
            "relevance_score":
            5,
            "relevance_reason":
            "The 'Star Wars and Related Entities' community offers highly engaging, action-packed movies that are ideal for a group setting. The franchise's mix of action, adventure, and strong characters make it a perfect fit for a fun movie session with friends.",
            "key_movies": [
                "Star Wars: Episode IV - A New Hope",
                "Star Wars: Episode V - The Empire Strikes Back",
                "Star Wars: Episode VI - Return of the Jedi"
            ],
            "key_theme":
            ["Adventure", "Sci-fi", "Good vs. Evil", "Heroic journeys"]
        }, {
            "community_id":
            36,
            "relevance_score":
            4,
            "relevance_reason":
            "The 'Movies and Characters Network' provides a diverse range of movies and interconnected stories which are ideal for group discussions and engagement. The variety in genres ensures that there's something for everyone, making it suitable for a fun group watch.",
            "key_movies": ["Leon: The Professional", "JFK", "Blade Runner"],
            "key_themes": [
                "Interconnectedness", "Thematic diversity",
                "Director-actor collaborations"
            ]
        }, {
            "community_id":
            41,
            "relevance_score":
            4,
            "relevance_reason":
            "The 'Intrigue in Movies - Sherlock Holmes, Nine Queens, and J\u00fbb\u00ea ninp\u00fbch\u00f4' community offers movies with engaging and complex plots that can be fun and intellectually stimulating for a group to watch together. Mysteries and thrillers are often popular in group settings.",
            "key_movies":
            ["Sherlock Holmes", "Nine Queens", "Vampire Hunter D: Bloodlust"],
            "key_themes": ["Mystery", "Suspense", "Action"]
        }],
        "irrelevant_communities": ["30", "34", "35", "37", "38"]
    }, {
        "query_analysis":
        "The user is looking for a fun and engaging movie session with friends (referred to as 'bros') for the weekend, implying a preference for entertaining, possibly action-packed or comedic films suitable for a group setting.",
        "query_decomposition": [{
            "query":
            "fun movie session",
            "explanation":
            "The user wants a movie that promises entertainment and enjoyment."
        }, {
            "query":
            "with my bros",
            "explanation":
            "The movie should be suitable for a group of male friends, indicating a preference for genres and themes that typically appeal to this demographic, such as action, comedy, or adventure."
        }, {
            "query":
            "weekend session",
            "explanation":
            "The movie should be engaging and hold the group's attention, making the weekend memorable."
        }],
        "final_relevant_communities": [{
            "community_id":
            48,
            "relevance_score":
            5,
            "relevance_reason":
            "The 'James Bond' franchise offers a mix of action, adventure, and iconic characters, perfectly suited for a fun and thrilling group movie session.",
            "key_movies": ["Skyfall", "Spectre", "Casino Royale"],
            "key_theme": ["Espionage", "Action", "Thrilling adventures"]
        }, {
            "community_id":
            43,
            "relevance_score":
            4,
            "relevance_reason":
            "The 'Influential Movie Personalities Network' features dynamic and iconic films like 'The Matrix' and 'Back to the Future,' which are renowned for their action-packed and entertaining nature.",
            "key_movies": ["The Matrix", "Back to the Future", "The Departed"],
            "key_theme": ["Science fiction", "Action", "Dynamic storytelling"]
        }, {
            "community_id":
            46,
            "relevance_score":
            4,
            "relevance_reason":
            "The 'Movies and Characters Relationship Analysis' community's diverse range of genres and notable films include 'Django Unchained' and 'Rocky,' offering engaging stories and action.",
            "key_movies": ["Django Unchained", "Rocky", "Creed"],
            "key_theme": ["Action", "Drama", "Engaging fight scenes"]
        }],
        "irrelevant_communities": ["39", "42", "40", "45", "47"]
    }, {
        "query_analysis":
        "The query indicates a desire for an enjoyable group viewing experience, likely filled with engaging, exciting, or amusing content suitable for a group of male friends on a weekend. Implied preferences include action, humor, or high-energy entertainment that can be enjoyed collectively.",
        "query_decomposition": [{
            "query":
            "fun weekend movie session",
            "explanation":
            "The main requirement is for the movie session to be fun and enjoyable, implying genres such as comedy, action, or adventure that ensure entertainment."
        }, {
            "query":
            "movie session with my bros",
            "explanation":
            "The term 'bros' suggests a group of male friends, likely preferring movies with themes of friendship, camaraderie, action, or light-hearted content."
        }],
        "final_relevant_communities": [{
            "community_id":
            53,
            "relevance_score":
            5,
            "relevance_reason":
            "The Indonesian Action Movie Community with films like 'The Raid' and 'The Raid 2' offers high-energy action that is perfectly suited for a group of friends looking for exciting and thrilling entertainment.",
            "key_movies": ["The Raid", "The Raid 2"],
            "key_theme": ["Action", "Friendship", "Martial Arts"]
        }, {
            "community_id":
            49,
            "relevance_score":
            4,
            "relevance_reason":
            "The Interconnected Movies and Filmmakers community features films like 'Beasts of No Nation' and 'Sicario' which, while having serious themes, are action-packed and can be engaging for a group viewing session. ",
            "key_movies": ["Beasts of No Nation", "Sicario"],
            "key_theme": ["Action", "Conflict", "Collaboration"]
        }, {
            "community_id":
            51,
            "relevance_score":
            3,
            "relevance_reason":
            "The French Thriller Movie Community includes suspenseful and gripping movies that might appeal to a group of friends looking for intense and thrilling films.",
            "key_movies": ["The Wages of Fear", "Les Diaboliques"],
            "key_theme": ["Thriller", "Suspense", "Psychological"]
        }],
        "irrelevant_communities": ["44", "50", "52", "57", "58"]
    }, {
        "query_analysis":
        "The user's query indicates they are planning a weekend movie session with friends. The emphasis on 'fun' suggests they are looking for movies that are entertaining, which could mean light-hearted genres like comedies, action, possibly adventure, or popular blockbusters, rather than heavy dramas or films with intense emotional content. Since it is a group of 'bros,' there might also be an implicit preference for movies that appeal to a typical male audience, which could include action-packed films or those with themes of camaraderie.",
        "query_decomposition": [{
            "query":
            "Fun movies for a group of friends",
            "explanation":
            "The primary aspect of the query is finding movies that are enjoyable and suitable for a group setting, likely implying genres that provoke laughter, excitement, or relaxed enjoyment."
        }, {
            "query":
            "Movies with themes of camaraderie and brotherhood",
            "explanation":
            "Given that the user's audience includes 'bros,' films that focus on themes of friendship, teamwork, or close-knit relations could be particularly engaging."
        }, {
            "query":
            "Appealing genres for a male audience",
            "explanation":
            "To cater to a group of 'bros,' genres like action, comedy, adventure, or even some well-crafted horror or thriller movies might be especially appealing."
        }],
        "final_relevant_communities": [{
            "community_id":
            62,
            "relevance_score":
            5,
            "relevance_reason":
            "The 'Casino' community focuses on themes of power, betrayal, and intense interpersonal dynamics, which could provide an exciting and engaging movie experience for a group of friends looking for a thrilling film.",
            "key_movies": ["Casino"],
            "key_theme": ["Greed", "Power", "Betrayal", "High stakes"]
        }, {
            "community_id":
            56,
            "relevance_score":
            4,
            "relevance_reason":
            "The 'Butterfly Effect' community delves into the complexities of supernatural powers and time travel, offering a thought-provoking yet entertaining movie experience.",
            "key_movies": ["The Butterfly Effect"],
            "key_theme":
            ["Supernatural powers", "Time travel", "Alternate realities"]
        }, {
            "community_id":
            61,
            "relevance_score":
            3,
            "relevance_reason":
            "'A Man Called Ove' is a comedy-drama that blends elements of humor and emotion, which might still be appealing to the group, especially because of its heartfelt narrative.",
            "key_movies": ["A Man Called Ove"],
            "key_theme": ["Friendship", "Loss", "Redemption", "Humor"]
        }],
        "irrelevant_communities": ["54", "55", "59", "60", "63"]
    }, {
        "query_analysis":
        "The user is seeking to enjoy a light-hearted and entertaining movie time with friends, implying preferences for movies that are fun, engaging, and possibly comedic or action-packed rather than serious or artistic.",
        "query_decomposition": [{
            "query":
            "fun weekend movie",
            "explanation":
            "The user wants an enjoyable and entertaining movie suitable for a relaxed setting like a weekend."
        }, {
            "query":
            "with bros",
            "explanation":
            "The viewing group is close friends, which implies movies that promote camaraderie, humor, and possibly action or sports themes which are often enjoyed in male friend groups."
        }],
        "final_relevant_communities": [{
            "community_id":
            75,
            "relevance_score":
            5,
            "relevance_reason":
            "'My Sassy Girl' is a romantic comedy with a significant cultural impact, providing a fun and entertaining watch suitable for a relaxed group of friends.",
            "key_movies": ["My Sassy Girl"],
            "keyThemes":
            ["romantic comedy", "character interactions", "cultural impact"]
        }, {
            "community_id":
            83,
            "relevance_score":
            4,
            "relevance_reason":
            "The 'Aliens' franchise offers thrilling sci-fi action which can be a fun group experience, especially for friends who enjoy intense and adventurous movies.",
            "key_movies": ["Aliens"],
            "keyThemes": ["sci-fi", "adventure", "action"]
        }, {
            "community_id":
            77,
            "relevance_score":
            4,
            "relevance_reason":
            "'D-Day' provides a historically impactful film with action and drama, possibly appealing to a group of friends with an interest in war movies.",
            "key_movies": ["D-Day"],
            "keyThemes": ["historical", "action", "drama"]
        }],
        "irrelevant_communities": ["71", "70", "76", "74", "78"]
    }, {
        "query_analysis":
        "The user is looking to organize a fun movie session with friends, implying a preference for movies that are enjoyable, possibly light-hearted, action-packed, or comedic. The goal is entertainment and possibly a shared fun experience with friends.",
        "query_decomposition": [{
            "query":
            "Fun movies for a weekend session",
            "explanation":
            "The primary goal is to find movies that are entertaining and suitable for a laid-back weekend with friends. This includes genres like action, comedy, or adventure."
        }, {
            "query":
            "Movies suitable for watching with friends (bros)",
            "explanation":
            "The content should be suitable for a group of male friends, which might imply a preference for 'bro' movies - films with themes of camaraderie, action, or comedy."
        }],
        "final_relevant_communities": [{
            "community_id":
            73,
            "relevance_score":
            5,
            "relevance_reason":
            "Bruce Lee and Robert Clouse's iconic martial arts films are action-packed and can be highly entertaining for a group of friends. 'Enter the Dragon' and 'The Game of Death' are classics that offer thrilling action and memorable fight scenes, making them great choices for a fun weekend session.",
            "key_movies": ["Enter the Dragon", "The Game of Death"],
            "key_themes": ["martial arts", "action", "iconic fights"]
        }, {
            "community_id":
            69,
            "relevance_score":
            4,
            "relevance_reason":
            "'The Man from Earth' offers an intriguing and thought-provoking narrative with strong ensemble dynamics, which can be enjoyable for a group of friends who appreciate deep discussions and intellectual stimulation.",
            "key_movies": ["The Man from Earth"],
            "key_themes":
            ["intellectual narrative", "discussion-worthy", "ensemble cast"]
        }, {
            "community_id":
            66,
            "relevance_score":
            3,
            "relevance_reason":
            "'Four Men' has a strong collaborative element and notable performances, which might be interesting for a group of friends looking for a well-crafted film with good acting.",
            "key_movies": ["Four Men"],
            "key_themes": ["collaboration", "strong performances"]
        }],
        "irrelevant_communities": ["65", "67", "68", "64", "72"]
    }, {
        "query_analysis":
        "The user is looking for movies to enjoy with friends over a weekend, suggesting a preference for entertaining, engaging, and potentially light-hearted or action-packed films. They want a fun and enjoyable experience, which might lean towards genres like comedy, action, or adventure rather than heavy dramas or emotionally intense films.",
        "query_decomposition": [{
            "query":
            "fun weekend movie session",
            "explanation":
            "Identifies the setting and emotional context of the movie-watching experience, focusing on movies that would be enjoyable and entertaining in a social setting."
        }, {
            "query":
            "with my bros",
            "explanation":
            "Indicates a preference for movies that are suitable for a male group of friends, potentially implying genres that are stereotypically popular with male audiences such as action, comedy, sci-fi, or thrillers."
        }],
        "final_relevant_communities": [{
            "community_id":
            79,
            "relevance_score":
            5,
            "relevance_reason":
            "City of God showcases intense action, gripping storyline, and high energy which can be highly engaging and entertaining for a group of friends.",
            "key_movies": ["City of God"],
            "key_theme":
            ["urban violence", "action", "gritty realism", "ensemble cast"]
        }, {
            "community_id":
            82,
            "relevance_score":
            4,
            "relevance_reason":
            "Howl's Moving Castle offers a whimsical and fantastical adventure that can be a fun and visually stimulating experience for friends.",
            "key_movies": ["Howl's Moving Castle"],
            "key_theme": ["adventure", "fantasy", "character relationships"]
        }, {
            "community_id":
            84,
            "relevance_score":
            3,
            "relevance_reason":
            "The Holy Mountain, while visually striking and unique, may offer a different viewing experience that could be intriguing but may not be universally 'fun' for all groups.",
            "key_movies": ["The Holy Mountain"],
            "key_theme": ["surrealism", "symbolism", "avant-garde"]
        }, {
            "community_id":
            86,
            "relevance_score":
            2,
            "relevance_reason":
            "Das Boot, being a war movie with intense scenes and a focus on survival, might be less 'fun' but still highly engaging for some groups.",
            "key_movies": ["Das Boot"],
            "key_theme": ["war", "survival", "intensity"]
        }, {
            "community_id":
            85,
            "relevance_score":
            2,
            "relevance_reason":
            "The Intouchables is a heartwarming and uplifting film that, while not action-packed, offers a feel-good experience that can be enjoyed by friends.",
            "key_movies": ["The Intouchables"],
            "key_theme": ["friendship", "humor", "uplifting"]
        }],
        "irrelevant_communities": ["80", "81", "88"]
    }, {
        "query_analysis":
        "The user is looking to have a fun weekend movie session with friends, which likely indicates a preference for light-hearted, entertaining, and possibly action-packed movies that can be enjoyed in a social setting. The absence of specific genres, actors, or directors indicates an openness to various types of movies, as long as they are enjoyable in a group setting.",
        "query_decomposition": [{
            "query":
            "fun weekend",
            "explanation":
            "This suggests a preference for movies that are light-hearted, entertaining, and enjoyable in a social setting, likely including comedy, action, adventure, or feel-good movies."
        }, {
            "query":
            "movie session with bros",
            "explanation":
            "This sub-query emphasizes the social aspect of the movie session, suggesting that the movies should be appealing to a group of friends, possibly male, and should foster a fun, communal experience."
        }],
        "final_relevant_communities": [{
            "community_id":
            91,
            "relevance_score":
            5,
            "relevance_reason":
            "'Star Trek II: The Wrath of Khan' has a strong fan base and offers a mix of action, adventure, and nostalgia, which could be appealing for a group of friends looking for an engaging and fun movie session.",
            "key_movies": ["Star Trek II: The Wrath of Khan"],
            "key_themes": ["Action", "Adventure", "Nostalgia"]
        }, {
            "community_id":
            95,
            "relevance_score":
            4,
            "relevance_reason":
            "Although 'The General' is a silent film, Buster Keaton's comedic and action-packed performances can provide a unique and highly entertaining experience suitable for a group movie session, provided the group is open to classic cinema.",
            "key_movies": ["The General"],
            "key_themes": ["Comedy", "Action", "Classic Cinema"]
        }, {
            "community_id":
            92,
            "relevance_score":
            3,
            "relevance_reason":
            "'The Muppet Movie' offers light-hearted, family-friendly fun with a mix of comedy and music, and while it may not be the first choice for all adult groups, it can offer nostalgia and enjoyment in the right social context.",
            "key_movies": ["The Muppet Movie"],
            "key_themes": ["Comedy", "Family", "Nostalgia"]
        }],
        "irrelevant_communities": ["90", "87", "89", "93", "98"]
    }, {
        "query_analysis":
        "The user's query suggests a desire for a relaxed, enjoyable, and perhaps light-hearted movie-watching experience with friends. The context implies a fun and potentially communal atmosphere, where engaging, entertaining films would be ideal.",
        "query_decomposition": [{
            "query":
            "fun weekend movie",
            "explanation":
            "The primary request is for movies that contribute to a fun and enjoyable weekend."
        }, {
            "query":
            "movie session with bros",
            "explanation":
            "This implies a group setting with male friends, suggesting films that appeal to a male audience and prompt camaraderie."
        }],
        "final_relevant_communities": [{
            "community_id":
            97,
            "relevance_score":
            5,
            "relevance_reason":
            "Toy Story 2 is family-friendly, fun, and Woody's character with themes of friendship, loyalty would resonate well with a group of friends.",
            "key_movies": ["Toy Story 2"],
            "key_theme": ["Friendship", "Fun", "Adventure"]
        }, {
            "community_id":
            99,
            "relevance_score":
            4,
            "relevance_reason":
            "'Withnail & I' has dark humor and memorable characters, fitting well for a group of friends seeking a comedic yet meaningful experience.",
            "key_movies": ["Withnail & I"],
            "key_theme": ["Dark Humor", "Friendship", "Camaraderie"]
        }],
        "irrelevant_communities": ["96", "94"]
    }]
    output, usage = asyncio.run(queryReduce(llm, user_query, mapOutputs))
    print(f"USAGE: ", usage)
    print(f"OUTPUT: \n{json.dumps(output, indent=4)}")
