"""Utility functions for cleaning Enron email data."""
import re
from collections import namedtuple


# Functions for checking which sections of the text we are in

def check_for_match_factory(match_str, match_str_len):
    return lambda s: s[:match_str_len] == match_str

match_strs = ['Message-ID', 'Date', 'From', 'To', 'Subject', 'Mime-Version',
              'Content-Type', 'Content-Transfer-Encoding', 'X-From', 'X-To',
              'X-cc', 'X-bcc', 'X-Folder', 'X-Origin', 'X-FileName']

match_funcs = [check_for_match_factory(s, len(s)) for s in match_strs]

match_tup = namedtuple('Matcher', ['string', 'function'])
match_checks = [match_tup(s, f) for s, f in zip(match_strs, match_funcs)]

ignore_sections = ['Mime-Version', 'Content-Type', 'Content-Transfer-Encoding',
                   'X-Folder', 'X-Origin', 'X-FileName']

# Functions for cleaning individual sections of the structured data

def extract_message_id(text):
    return float('.'.join(text[13:].split('.')[:2]))

def extract_date(text):
    return text[6:]

def extract_from(text):
    return text[6:]

def extract_to(text):
    return text[4:].strip()

def extract_subject(text):
    return text[9:].strip()

def extract_mime_version(text):
    return ''

def extract_content_type(text):
    return ''

def extract_content_encoding(text):
    return ''

name_clean_list = ['"', "'", "\([^()]*\)", "\<[^<>]*\>", "[a-zA-Z]*@[a-zA-Z]*"]
name_clean_re = re.compile("|".join(name_clean_list))

def extract_from_name(text):
    return extract_name(text[8:])

def extract_to_name(text):
    return extract_name(text[6:])

def extract_cc_name(text):
    return extract_name(text[6:])

def extract_bcc_name(text):
    return extract_name(text[7:])

def extract_name(text):
    text.replace(' @ ', '@')
    return ', '.join([*map(lambda s: s.strip(),
                           name_clean_re.sub('', text).split(','))])

def extract_folder(text):
    return ''

def extract_origin(text):
    return ''

def extract_filename(text):
    return ''

clean_functions = {'Message-ID': extract_message_id,
                   'Date': extract_date,
                   'From': extract_from,
                   'To': extract_to,
                   'Subject': extract_subject,
                   'Mime-Version': extract_mime_version,
                   'Content-Type': extract_content_type,
                   'Content-Transfer-Encoding': extract_content_encoding,
                   'X-From': extract_from_name,
                   'X-To': extract_to_name,
                   'X-cc': extract_cc_name,
                   'X-bcc': extract_bcc_name,
                   'X-Folder': extract_folder,
                   'X-Origin': extract_origin,
                   'X-FileName': extract_filename}

def clean_entry(orig_i, row):
    """Clean the entire entry for a row in the Enron dataset."""
    lines = row['message'].split('\n')

    results_dict = {}

    section_id = 0
    match_func = match_checks[section_id].function
    curr_section_name = match_checks[section_id].string
    last_section_name = ''
    for i, line in enumerate(lines):
        if match_func(line):
            # Store the data for the last section we hit
            if section_id != 0 and last_section_name not in ignore_sections:
                clean_f = clean_functions[last_section_name]
                results_dict[last_section_name] = clean_f(curr_section)
            # Extract the current section
            curr_section = line.strip()
            # Increment the section we are checking for
            last_section_name = curr_section_name
            section_id += 1
            try:
                match_func = match_checks[section_id].function
                curr_section_name = match_checks[section_id].string
            except IndexError:
                # We've reached the end of the structured data
                break
        else:
            # Append the current line to the current section
            curr_section += ' ' + line.strip()

    # Add in the email contents
    results_dict['raw_text'] = ''.join(lines[i+1:]).strip()
    results_dict['orig_i'] = orig_i
    results_dict['orig_fname'] = row['file']
    results_dict['fname_prefix'] = row['file'].split('/')[0]
    return results_dict['Message-ID'], results_dict