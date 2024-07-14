'''Community Report for community chunks'''
import random
import asyncio
import time
import json_repair as json
from typing import List, Dict
from llm.localllm import LocalLLM
from graphragrec.prompts.prompts import EMBED
from graphragrec.utils.usage import calculateUsages
from graphragrec.schemas.embed.community import CommunityReportTool


async def communityReport(llm: LocalLLM, model: str,
                          community_data: List[Dict]):
    # time.sleep(random.choice([2, 3, 4, 5, 6, 10, 12]))
    message_content = f"```{community_data}```"
    messages = [{
        "role": "system",
        "content": EMBED.COMMUNITY
    }, {
        "role": "user",
        "content": message_content
    }]
    output, usage = await llm.__function_call__(
        messages,
        model,
        CommunityReportTool.tools,
        tool_choice=CommunityReportTool.tool_choice,
        max_tokens=4000)
    return output, usage


if __name__ == "__main__":
    import json
    from configs import OPENAI_API_KEY
    llm = LocalLLM(api_key=OPENAI_API_KEY)
    community_data = [{
        "entity": "Frank Capra",
        "relations-claims": {
            "It's a Wonderful Life": {
                "type":
                "DIRECTED_BY",
                "source":
                "It's a Wonderful Life",
                "target":
                "Frank Capra",
                "description":
                "Frank Capra is the director of the movie 'It's a Wonderful Life.'",
                "strength":
                9,
                "claims": [{
                    "predicate": "DIRECTED_BY",
                    "value": "",
                    "confidence": 90
                }]
            },
            "Mr. Smith Goes to Washington": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Frank Capra",
                "target":
                "Movie",
                "type":
                "DIRECTED_BY",
                "description":
                "Directed the movie",
                "strength":
                9,
                "claims": [{
                    "predicate": "DIRECTED_BY",
                    "value": "",
                    "confidence": 100
                }]
            },
            "It Happened One Night": {
                "type":
                "DIRECTED_BY",
                "source":
                "It Happened One Night",
                "target":
                "Frank Capra",
                "description":
                "Frank Capra directed the movie 'It Happened One Night'",
                "strength":
                9,
                "claims": [{
                    "predicate": "DIRECTED_BY",
                    "value": "",
                    "confidence": 90
                }]
            },
            "Arsenic and Old Lace": {
                "type":
                "DIRECTED_BY",
                "source":
                "Frank Capra",
                "target":
                "Arsenic and Old Lace",
                "description":
                "Frank Capra directed the movie 'Arsenic and Old Lace'.",
                "strength":
                10,
                "claims": [{
                    "predicate": "DIRECTED_BY",
                    "value": "",
                    "confidence": 100
                }]
            }
        }
    }, {
        "entity": "Claude Rains",
        "relations-claims": {
            "Casablanca": {
                "type":
                "STARS_IN",
                "source":
                "Casablanca",
                "target":
                "Claude Rains",
                "description":
                "Claude Rains starred in the movie Casablanca.",
                "strength":
                8,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 100
                }]
            },
            "Mr. Smith Goes to Washington": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Claude Rains",
                "target":
                "Movie",
                "type":
                "STARS_IN",
                "description":
                "Stars in the movie",
                "strength":
                7,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 100
                }]
            },
            "The Invisible Man": {
                "type":
                "ACTED_IN",
                "source":
                "The Invisible Man",
                "target":
                "Claude Rains",
                "description":
                "Claude Rains played a role in The Invisible Man",
                "strength":
                8,
                "claims": [{
                    "predicate": "ACTED_IN",
                    "value": "",
                    "confidence": 90
                }]
            }
        }
    }, {
        "entity": "2016",
        "relations-claims": {
            "Kimi no na wa.": {
                "type": "HAS_ENTITY"
            },
            "Two strangers": {
                "source":
                "Two strangers",
                "target":
                "2016",
                "type":
                "RELEASED_IN",
                "description":
                "The movie \"Two strangers\" was released in 2016",
                "strength":
                9,
                "claims": [{
                    "predicate": "RELEASED_IN",
                    "value": "",
                    "confidence": 100
                }]
            },
            "Contratiempo": {
                "type": "HAS_ENTITY"
            },
            "Oriol Paulo": {
                "source": "2016",
                "target": "Oriol Paulo",
                "type": "RELEASED_IN",
                "description":
                "The movie was released in the year 2016 and directed by Oriol Paulo.",
                "strength": 10
            },
            "Hacksaw Ridge": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "2016",
                "target":
                "Movie",
                "type":
                "RELEASED_IN",
                "description":
                "The year the movie was released",
                "strength":
                9,
                "claims": [{
                    "predicate": "RELEASE_YEAR",
                    "value": "",
                    "confidence": 100
                }, {
                    "predicate": "RELEASED_IN",
                    "value": "",
                    "confidence": 95
                }]
            },
            "Deadpool": {
                "type": "HAS_ENTITY"
            },
            "Ryan Reynolds": {
                "source": "Ryan Reynolds",
                "target": "2016",
                "type": "RELEASED_IN",
                "description":
                "The movie was released in 2016 and Ryan Reynolds is part of the cast",
                "strength": 7
            },
            "R": {
                "source": "R",
                "target": "2016",
                "type": "RELEASED_IN",
                "description":
                "The movie was released in 2016 with a Restricted rating",
                "strength": 6
            },
            "Hunt for the Wilderpeople": {
                "type": "HAS_ENTITY"
            },
            "Sing Street": {
                "type": "HAS_ENTITY"
            }
        }
    }, {
        "entity": "Gael Garc\u00eda Bernal",
        "relations-claims": {
            "Coco": {
                "type": "HAS_ENTITY"
            },
            "Music Film": {
                "source": "Gael Garc\u00eda Bernal",
                "target": "Music Film",
                "type": "ACTED_IN",
                "description": "Gael Garc\u00eda Bernal acted in the movie",
                "strength": 6
            },
            "Amores perros": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source": "Movie",
                "target": "Gael Garc\u00eda Bernal",
                "type": "ACTOR",
                "description": "One of the cast members of the movie.",
                "strength": 7
            },
            "Diarios de motocicleta": {
                "type": "HAS_ENTITY"
            },
            "The Motorcycle Diaries": {
                "source":
                "Gael Garc\u00eda Bernal",
                "target":
                "The Motorcycle Diaries",
                "type":
                "STARS_IN",
                "description":
                "Actor who is part of the cast of The Motorcycle Diaries",
                "strength":
                8,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 100
                }]
            },
            "Y tu mam\u00e1 tambi\u00e9n": {
                "type": "HAS_ENTITY"
            },
            "Y Tu Mam\u00e1 Tambi\u00e9n": {
                "source":
                "Y Tu Mam\u00e1 Tambi\u00e9n",
                "target":
                "Gael Garc\u00eda Bernal",
                "type":
                "STARS_IN",
                "description":
                "Gael Garc\u00eda Bernal stars in the movie Y Tu Mam\u00e1 Tambi\u00e9n.",
                "strength":
                8,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 100
                }]
            }
        }
    }, {
        "entity": "Hrishikesh Mukherjee",
        "relations-claims": {
            "Anand": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Hrishikesh Mukherjee",
                "target":
                "Movie",
                "type":
                "DIRECTED_BY",
                "description":
                "The director of the movie.",
                "strength":
                9,
                "claims": [{
                    "predicate": "DIRECTED_BY",
                    "value": "",
                    "confidence": 95
                }]
            }
        }
    }, {
        "entity": "Rajesh Khanna",
        "relations-claims": {
            "Anand": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Rajesh Khanna",
                "target":
                "Movie",
                "type":
                "STARS_IN",
                "description":
                "One of the cast members of the movie.",
                "strength":
                7,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 80
                }]
            }
        }
    }, {
        "entity": "Amitabh Bachchan",
        "relations-claims": {
            "Anand": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Amitabh Bachchan",
                "target":
                "Movie",
                "type":
                "STARS_IN",
                "description":
                "One of the cast members of the movie.",
                "strength":
                7,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 80
                }]
            },
            "Black": {
                "type": "HAS_ENTITY"
            },
            "The cathartic tale of a young woman who can't see, hear or talk and the teacher who brings a ray of light into her dark world":
            {
                "source":
                "The cathartic tale of a young woman who can't see, hear or talk and the teacher who brings a ray of light into her dark world",
                "target": "Amitabh Bachchan",
                "type": "STARS_IN",
                "description": "Amitabh Bachchan starred in the movie",
                "strength": 8,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 8
                }]
            },
            "Sholay": {
                "type": "HAS_ENTITY"
            },
            "Pink": {
                "type": "HAS_ENTITY"
            },
            "untitled movie": {
                "source":
                "untitled movie",
                "target":
                "Amitabh Bachchan",
                "type":
                "STARS_IN",
                "description":
                "An actor who is part of the cast",
                "strength":
                8,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 100
                }]
            }
        }
    }, {
        "entity": "Sumita Sanyal",
        "relations-claims": {
            "Anand": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Sumita Sanyal",
                "target":
                "Movie",
                "type":
                "STARS_IN",
                "description":
                "One of the cast members of the movie.",
                "strength":
                7,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 80
                }]
            }
        }
    }, {
        "entity": "Ramesh Deo",
        "relations-claims": {
            "Anand": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Ramesh Deo",
                "target":
                "Movie",
                "type":
                "STARS_IN",
                "description":
                "One of the cast members of the movie.",
                "strength":
                7,
                "claims": [{
                    "predicate": "STARS_IN",
                    "value": "",
                    "confidence": 80
                }]
            }
        }
    }, {
        "entity":
        "A man goes to extreme lengths to save his family from punishment after the family commits an accidental crime",
        "relations-claims": {
            "Drishyam": {
                "type": "HAS_ENTITY"
            }
        }
    }, {
        "entity": "Mohanlal",
        "relations-claims": {
            "Drishyam": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Movie",
                "target":
                "Mohanlal",
                "type":
                "ACTOR_IN",
                "description":
                "Mohanlal is part of the cast of the movie.",
                "strength":
                9,
                "claims": [{
                    "predicate": "ACTOR_IN",
                    "value": "",
                    "confidence": 90
                }]
            }
        }
    }, {
        "entity": "Meena",
        "relations-claims": {
            "Drishyam": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Movie",
                "target":
                "Meena",
                "type":
                "ACTOR_IN",
                "description":
                "Meena is part of the cast of the movie.",
                "strength":
                8,
                "claims": [{
                    "predicate": "ACTOR_IN",
                    "value": "",
                    "confidence": 80
                }]
            }
        }
    }, {
        "entity": "Asha Sharath",
        "relations-claims": {
            "Drishyam": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Movie",
                "target":
                "Asha Sharath",
                "type":
                "ACTOR_IN",
                "description":
                "Asha Sharath is part of the cast of the movie.",
                "strength":
                7,
                "claims": [{
                    "predicate": "ACTOR_IN",
                    "value": "",
                    "confidence": 70
                }]
            }
        }
    }, {
        "entity": "Ansiba",
        "relations-claims": {
            "Drishyam": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Movie",
                "target":
                "Ansiba",
                "type":
                "ACTOR_IN",
                "description":
                "Ansiba is part of the cast of the movie.",
                "strength":
                7,
                "claims": [{
                    "predicate": "ACTOR_IN",
                    "value": "",
                    "confidence": 70
                }]
            }
        }
    }, {
        "entity": "Jeethu Joseph",
        "relations-claims": {
            "Drishyam": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Movie",
                "target":
                "Jeethu Joseph",
                "type":
                "DIRECTED_BY",
                "description":
                "The movie is directed by Jeethu Joseph.",
                "strength":
                10,
                "claims": [{
                    "predicate": "DIRECTED_BY",
                    "value": "",
                    "confidence": 100
                }]
            }
        }
    }, {
        "entity": "Elem Klimov",
        "relations-claims": {
            "Idi i smotri": {
                "type": "HAS_ENTITY"
            },
            "Movie": {
                "source":
                "Elem Klimov",
                "target":
                "Movie",
                "type":
                "DIRECTED_BY",
                "description":
                "Elem Klimov directed the movie",
                "strength":
                9,
                "claims": [{
                    "predicate": "DIRECTED_BY",
                    "value": "",
                    "confidence": 9
                }]
            }
        }
    }]

    output, usage = asyncio.run(
        communityReport(llm, "gpt-3.5-turbo-0125", community_data))
    print("USAGE: ", usage)
    print(json.dumps(output, indent=4))
