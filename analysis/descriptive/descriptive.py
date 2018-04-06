def parse_time_value(time_string):
    # Parse the time string value and interpret it in seconds
    try:
        time_split = time_string.split(':')
        return int(time_split[0]) * 60.0 + float(time_split[1])
    except AttributeError:
        return time_string
