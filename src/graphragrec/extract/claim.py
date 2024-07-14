import asyncio
import json_repair as json
from typing import List, Dict
from llm.localllm import LocalLLM
from graphragrec.prompts.prompts import EXTRACT
from graphragrec.utils.usage import calculateUsages
from graphragrec.schemas.extract.claims import ClaimTool


async def extractEntityClaims(llm: LocalLLM, model: str, movie_overview: str,
                              entities: List[Dict]):
    message_content = f"Movie: ``{movie_overview}``\n Entities: ```{entities}```"
    messages = [{
        "role": "system",
        "content": EXTRACT.CLAIM
    }, {
        "role": "user",
        "content": f"Claim for Entities. {message_content}"
    }]
    # output, usage = await llm.__complete__(messages, model)
    output, usage = await llm.__function_call__(
        messages, model, ClaimTool.tools, tool_choice=ClaimTool.tool_choice)

    # claims = json.loads(output)
    claims = output.get("claims")
    return claims, usage


async def extractRelationClaim(llm: LocalLLM, model: str, movie_overview: str,
                               relations: List[Dict]):
    message_content = f"Movie: ``{movie_overview}``\n Relations: ```{relations}```"
    messages = [{
        "role": "system",
        "content": EXTRACT.CLAIM
    }, {
        "role": "user",
        "content": f"Claim for Relations. {message_content}"
    }]
    # output, usage = await llm.__complete__(messages, model)
    output, usage = await llm.__function_call__(
        messages, model, ClaimTool.tools, tool_choice=ClaimTool.tool_choice)

    # claims = json.loads(output)
    claims = output.get("claims")
    return claims, usage


async def extractClaims(llm: LocalLLM, model: str, movie_overview: str,
                        entities: List[Dict], relations: List[Dict]):
    results = await asyncio.gather(*[
        extractEntityClaims(llm, model, movie_overview, entities),
        extractRelationClaim(llm, model, movie_overview, relations)
    ])
    entityClaims, relationClaims = results
    usages = calculateUsages([entityClaims[-1], relationClaims[-1]])
    return {"entity": entityClaims[0], "relation": relationClaims[0]}, usages


if __name__ == "__main__":
    import asyncio
    from configs import OPENAI_API_KEY
    llm = LocalLLM(api_key=OPENAI_API_KEY)
    movie_overview = "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.\nYear: 1994\nDirector: Frank Darabont\nCast: ['Tim Robbins', 'Morgan Freeman', 'Bob Gunton', 'William Sadler']\nCertificate: A"
    entities = [{
        'name':
        'The Shawshank Redemption',
        'type':
        'Movie',
        'id':
        '1',
        'description':
        'A drama film about two imprisoned men who develop a deep bond.'
    }, {
        'name': 'Frank Darabont',
        'type': 'Director',
        'id': '2',
        'description': 'Director of The Shawshank Redemption.'
    }, {
        'name': 'Tim Robbins',
        'type': 'Actor',
        'id': '3',
        'description': 'Actor in The Shawshank Redemption.'
    }, {
        'name': 'Morgan Freeman',
        'type': 'Actor',
        'id': '4',
        'description': 'Actor in The Shawshank Redemption.'
    }, {
        'name': 'Bob Gunton',
        'type': 'Actor',
        'id': '5',
        'description': 'Actor in The Shawshank Redemption.'
    }, {
        'name': 'William Sadler',
        'type': 'Actor',
        'id': '6',
        'description': 'Actor in The Shawshank Redemption.'
    }, {
        'name':
        'The Shawshank Redemption',
        'type':
        'Movie',
        'id':
        '1',
        'description':
        'A drama film about two imprisoned men who develop a deep bond and find solace and eventual redemption through acts of common decency.'
    }, {
        'name': 'Frank Darabont',
        'type': 'Director',
        'id': '2',
        'description': 'Director of The Shawshank Redemption.'
    }, {
        'name': 'Tim Robbins',
        'type': 'Actor',
        'id': '3',
        'description': 'Actor in The Shawshank Redemption.'
    }, {
        'name': 'Morgan Freeman',
        'type': 'Actor',
        'id': '4',
        'description': 'Actor in The Shawshank Redemption.'
    }, {
        'name': 'Bob Gunton',
        'type': 'Actor',
        'id': '5',
        'description': 'Actor in The Shawshank Redemption.'
    }, {
        'name': 'William Sadler',
        'type': 'Actor',
        'id': '6',
        'description': 'Actor in The Shawshank Redemption.'
    }, {
        'name':
        '1994',
        'type':
        'Year',
        'id':
        '7',
        'description':
        'The year The Shawshank Redemption was released.'
    }, {
        'name': 'A',
        'type': 'Certificate',
        'id': '8',
        'description': 'The MPAA rating for The Shawshank Redemption.'
    }]
    relations = [{
        'source': 'The Shawshank Redemption',
        'target': 'Frank Darabont',
        'type': 'DIRECTED_BY',
        'description':
        'Frank Darabont directed the movie The Shawshank Redemption.',
        'strength': 10
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Tim Robbins',
        'type': 'STARS_IN',
        'description':
        'Tim Robbins is an actor in the movie The Shawshank Redemption.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Morgan Freeman',
        'type': 'STARS_IN',
        'description':
        'Morgan Freeman is an actor in the movie The Shawshank Redemption.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Bob Gunton',
        'type': 'STARS_IN',
        'description':
        'Bob Gunton is an actor in the movie The Shawshank Redemption.',
        'strength': 7
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'William Sadler',
        'type': 'STARS_IN',
        'description':
        'William Sadler is an actor in the movie The Shawshank Redemption.',
        'strength': 6
    }, {
        'source': 'The Shawshank Redemption',
        'target': '1994',
        'type': 'RELEASED_IN',
        'description':
        'The Shawshank Redemption was released in the year 1994.',
        'strength': 9
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'A',
        'type': 'RECEIVED_RATING',
        'description':
        'The Shawshank Redemption received an A rating from the MPAA.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Frank Darabont',
        'type': 'DIRECTED_BY',
        'description':
        'Frank Darabont directed the movie The Shawshank Redemption.',
        'strength': 10
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Tim Robbins',
        'type': 'STARS_IN',
        'description':
        'Tim Robbins is an actor in the movie The Shawshank Redemption.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Morgan Freeman',
        'type': 'STARS_IN',
        'description':
        'Morgan Freeman is an actor in the movie The Shawshank Redemption.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Bob Gunton',
        'type': 'STARS_IN',
        'description':
        'Bob Gunton is an actor in the movie The Shawshank Redemption.',
        'strength': 7
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'William Sadler',
        'type': 'STARS_IN',
        'description':
        'William Sadler is an actor in the movie The Shawshank Redemption.',
        'strength': 6
    }, {
        'source': 'The Shawshank Redemption',
        'target': '1994',
        'type': 'RELEASED_IN',
        'description':
        'The Shawshank Redemption was released in the year 1994.',
        'strength': 9
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'A',
        'type': 'RECEIVED_RATING',
        'description':
        'The Shawshank Redemption received an A rating from the MPAA.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Movie',
        'type': 'HAS_GENRE',
        'description':
        'The Shawshank Redemption is a movie in the Drama genre.',
        'strength': 9
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Frank Darabont',
        'type': 'DIRECTED_BY',
        'description':
        'Frank Darabont directed the movie The Shawshank Redemption.',
        'strength': 10
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Tim Robbins',
        'type': 'STARS_IN',
        'description':
        'Tim Robbins is an actor in the movie The Shawshank Redemption.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Morgan Freeman',
        'type': 'STARS_IN',
        'description':
        'Morgan Freeman is an actor in the movie The Shawshank Redemption.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Bob Gunton',
        'type': 'STARS_IN',
        'description':
        'Bob Gunton is an actor in the movie The Shawshank Redemption.',
        'strength': 7
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'William Sadler',
        'type': 'STARS_IN',
        'description':
        'William Sadler is an actor in the movie The Shawshank Redemption.',
        'strength': 6
    }, {
        'source': 'The Shawshank Redemption',
        'target': '1994',
        'type': 'RELEASED_IN',
        'description':
        'The Shawshank Redemption was released in the year 1994.',
        'strength': 9
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'A',
        'type': 'RECEIVED_RATING',
        'description':
        'The Shawshank Redemption received an A rating from the MPAA.',
        'strength': 8
    }, {
        'source': 'The Shawshank Redemption',
        'target': 'Drama',
        'type': 'HAS_GENRE',
        'description': 'The Shawshank Redemption is a drama movie.',
        'strength': 9
    }, {
        'source': 'Tim Robbins',
        'target': 'Andy Dufresne',
        'type': 'PLAYS_CHARACTER',
        'description':
        'Tim Robbins plays Andy Dufresne in The Shawshank Redemption.',
        'strength': 9
    }, {
        'source': 'Morgan Freeman',
        'target': "Ellis 'Red' Redding",
        'type': 'PLAYS_CHARACTER',
        'description':
        "Morgan Freeman plays Ellis 'Red' Redding in The Shawshank Redemption.",
        'strength': 9
    }, {
        'source': 'Bob Gunton',
        'target': 'Ward Norton',
        'type': 'PLAYS_CHARACTER',
        'description':
        'Bob Gunton plays Ward Norton in The Shawshank Redemption.',
        'strength': 7
    }, {
        'source': 'William Sadler',
        'target': 'Heywood',
        'type': 'PLAYS_CHARACTER',
        'description':
        'William Sadler plays Heywood in The Shawshank Redemption.',
        'strength': 6
    }]
    output = asyncio.run(
        extractClaims(llm, "gpt-3.5-turbo-0125", movie_overview, entities,
                      relations))
    print(output)
