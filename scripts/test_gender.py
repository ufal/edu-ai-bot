import edubot.remote_services
import yaml
config = yaml.load(open('configs/default_config.yaml', 'r'), Loader=yaml.loader.SafeLoader)
RSH = edubot.remote_services.RemoteServiceHandler(config, set())
print(RSH.postprocess_gender('Jsem šťastná, že jsem tu mohla být.'))
print(RSH.postprocess_gender('Jsi šťastná?'))
print(RSH.postprocess_gender('Olga je šťastná.'))
print(RSH.postprocess_gender('Šla jsem a koupila pět rohlíků.'))
print(RSH.postprocess_gender('Jsem spisovatelka na volné noze.'))
