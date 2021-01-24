def path_or_name(string):
    kind = None
    if "/" in string:
        kind =  "path"
    else:
        kind = "name"

    return kind