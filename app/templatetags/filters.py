from django import template
import os

register = template.Library()

# Retrieve VideoTranscript CSV name
@register.filter
def basename(value):
    return os.path.basename(value)

# Make VideoURL into image
@register.filter
def split(value, arg):
    return value.split(arg)

# Make VideoURL into iframe
@register.filter
def replace_youtube_url(value, args):
    old, new = args.split(',')
    return value.replace(old.strip(), new.strip())