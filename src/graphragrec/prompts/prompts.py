class EXTRACT:
    ENTITIES = open("./graphragrec/prompts/extract/entity.txt").read()
    RELATION = open("./graphragrec/prompts/extract/relation.txt").read()
    CLAIM = open("./graphragrec/prompts/extract/claim.txt").read()


class EMBED:
    COMMUNITY = open("./graphragrec/prompts/embed/community.txt").read()
    COMBINE = open("./graphragrec/prompts/embed/communitycombine.txt").read()