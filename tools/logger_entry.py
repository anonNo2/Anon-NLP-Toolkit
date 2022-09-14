import logging
from pathlib import Path



def init_logger(log_file=None, log_file_level=logging.NOTSET,name=None):
    '''
    初始化日志
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

class GlobalLogger(object):
    '''
    中英对照日志类
    '''
    def __init__(self,language='zh', log_file=None, log_file_level=logging.NOTSET,name=None):
        self.language = language
        self.logger = init_logger(log_file, log_file_level,name)
        return
    def glo(self,chinese,eng=''):
        if self.language == 'zh':
            return chinese
        else:
            if eng:
                return eng
            else:
                return f'未集成英文日志(English logs are not integrated)-{chinese}'


    def info(self, zh_msg, eng_msg=''):
        if eng_msg:
            self.logger.info(self.glo(zh_msg,eng_msg))
        else:
            self.logger.info(zh_msg)
        return
    def warning(self, zh_msg, eng_msg=''):
        if eng_msg:
            self.logger.warning(self.glo(zh_msg,eng_msg))
        else:
            self.logger.warning(zh_msg)
        return
    
    def error(self, zh_msg, eng_msg=''):
        if eng_msg:
            self.logger.error(self.glo(zh_msg,eng_msg))
        else:
            self.logger.error(zh_msg)
        return
    def debug(self, zh_msg, eng_msg=''):
        if eng_msg:
            self.logger.debug(self.glo(zh_msg,eng_msg))
        else:
            self.logger.debug(zh_msg)
        return
    def critical(self, zh_msg, eng_msg=''):
        if eng_msg:
            self.logger.critical(self.glo(zh_msg,eng_msg))
        else:
            self.logger.critical(zh_msg)
        return
    def exception(self, zh_msg, eng_msg=''):
        if eng_msg:
            self.logger.exception(self.glo(zh_msg,eng_msg))
        else:
            self.logger.exception(zh_msg)
        return
    
    def no_glo(self,msg):
        self.logger.info(msg)

    def print_config(self,config):
        
        self.info("配置如下:\n","Running with the following configs:\n")
        param_print = ''
        for k, v in config.items():
            param_print += f"\t{k} : {str(v)}\n"
        self.no_glo("\n" + param_print + "\n")
        return
 




    
