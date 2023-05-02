import argparse
import os
import glob
import xml.etree.ElementTree as ET
import yaml
from yaml.loader import SafeLoader

from edubot.remote_services import RemoteServiceHandler


def process_file(inp_file, remote_service_handler, ofd, sep="\t"):
    print(inp_file)
    # extract all the contents from the <category> tags using xml parser
    # extract pairs from the <pattern> and <template> tags
    # save the contents to a list
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(inp_file, parser=parser)
    root = tree.getroot()
    for child in root:
        if child.tag == "category":
            pattern = child.find("pattern").text
            template = child.find("template").text
            if template is None or pattern is None:
                continue
            try:
                print(sep.join([pattern.lower().strip(),
                                template.lower().strip(),
                                remote_service_handler.translate_en2cs(pattern.lower().strip()).strip(),
                                remote_service_handler.translate_en2cs(template.lower().strip())]).strip(),
                      file=ofd, flush=True)
            except Exception as e:
                print(e)
                continue


def main(args):
    processed_files = ["alice.aiml", "update1.aiml", "inquiry.aiml", "history.aiml", "psychology.aiml", "mp4.aiml",
                       "astrology.aiml", "iu.aiml", "computers.aiml", "client.aiml", "reduction1.safe.aiml",
                       "religion.aiml", "drugs.aiml", "primeminister.aiml", "reductions-update.aiml",
                       "music.aiml", "numbers.aiml", "personality.aiml", "geography.aiml", "food.aiml",
                       "phone.aiml", "bot_profile.aiml", "ai.aiml", "reduction0.safe.aiml", "reduction2.safe.aiml",
                       "stories.aiml", "movies.aiml", "biography.aiml", "science.aiml", "salutations.aiml",
                       "reduction4.safe.aiml", "client_profile.aiml", "continuation.aiml", "money.aiml",
                       "emotion.aiml", "mp1.aiml", "knowledge.aiml", "mp0.aiml", "mp2.aiml", "default.aiml",
                       "sports.aiml","atomic.aiml", "reduction3.safe.aiml", "mp3.aiml", "mp6.aiml", "stack.aiml",
                       "bot.aiml", "date.aiml", "sex.aiml", "xfind.aiml", "psychology.aiml", "loebner10.aiml",
                       "reduction.names.aiml", "mp5.aiml"]
    processed_files.extend([f"{args.input}/{f}" for f in processed_files])
    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.aiml"))
    else:
        files = [args.input]
    with open(args.config, 'rt') as fd:
        config = yaml.load(fd, Loader=SafeLoader)
    remote_service_handler = RemoteServiceHandler(config)
    with open(args.output, 'at') as ofd:
        for inp_file in files:
            if inp_file in processed_files:
                continue
            process_file(inp_file, remote_service_handler, ofd, sep=args.separator)
            print("Done with file: ", inp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input file/directory")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--output", type=str, help="Path to output file")
    parser.add_argument("--separator", type=str, default="\t")
    args = parser.parse_args()
    main(args)
