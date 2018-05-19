# coding:utf-8

import logging


# 创建logger记录器

logger=logging.getLogger('Logfile')

# 设置日志显示级别

logger.setLevel(logging.WARNING)

# 创建日志处理器

sh=logging.FileHandler(filename='log.log')

# 创建日志格式：日志级别数值-日志级别名称-当前执行程序的路径-当前执行程序名称-日志的当前函数-日志当前的行号-日志的时间-进程id-打印线程id-线程名-日志信息
fmt='%(levelno)s-%(levelname)s-%(pathname)s-%(filename)s-%(funcName)s-%(lineno)d-%(asctime)s-%(process)d-%(thread)d-%(threadName)s-%(message)s'
datefmt = "%a %d %b %Y %H:%M:%S"
formatter=logging.Formatter(fmt=fmt,datefmt=datefmt)
sh.setFormatter(formatter)
logger.addHandler(sh)

# 自模块的导入要在日志配置完后面

logger.debug('debug message StreamHandler')
logger.info('info message StreamHandler')
logger.warning('warn message StreamHandler')
logger.error('error message StreamHandler')
logger.critical('critical message StreamHandler')



