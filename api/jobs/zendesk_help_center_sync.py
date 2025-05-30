#!/usr/bin/env python3
"""
Zendesk帮助中心文档同步定时任务

功能：
1. 同步指定语言的所有帮助中心文档（Articles）
2. 同步分类（Categories）和章节（Sections）结构
3. 下载文档内容和附件
4. 支持多语言内容同步
5. 保存到本地文件系统

使用方法：
1. 直接运行: python api/jobs/zendesk_help_center_sync.py --locale zh-cn
2. 定时任务: 配置cron job调用此脚本  
3. 作为模块导入: from api.jobs.zendesk_help_center_sync import ZendeskHelpCenterSync
"""

import sys
import os
import json
import logging
import requests
import schedule
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api import settings


class ZendeskHelpCenterSync:
    """Zendesk帮助中心文档同步类"""

    def __init__(self, subdomain: str, email: str, api_token: str):
        """
        初始化同步器
        
        Args:
            subdomain: Zendesk子域名
            email: 用户邮箱
            api_token: API Token
        """
        self.subdomain = subdomain
        self.email = email
        self.api_token = api_token
        self.base_url = f"https://{subdomain}.zendesk.com"
        self.base_json_data_dir = "data/zendesk_help_center/raw_data_json"
        self.base_markdown_data_dir = "data/zendesk_help_center/markdown_data"
        # 设置认证
        self.auth = (f"{email}/token", api_token)
        self.headers = {"Content-Type": "application/json"}

        # 设置日志
        self.setup_logging()

        # 验证连接
        if not self._test_connection():
            raise ValueError("无法连接到Zendesk API，请检查配置")

    def setup_logging(self):
        """设置日志记录"""
        # 创建logs目录（如果不存在）
        os.makedirs('logs', exist_ok=True)

        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/zendesk_help_center_sync_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _test_connection(self) -> bool:
        """测试API连接"""
        try:
            url = f"{self.base_url}/api/v2/help_center/categories.json"
            response = requests.get(url, auth=self.auth, headers=self.headers, timeout=10)

            if response.status_code == 200:
                self.logger.info("API连接测试成功")
                return True
            else:
                self.logger.error(f"API连接测试失败: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.exception(f"API连接测试异常: {e}")
            return False

    def get_categories(self, locale: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取所有分类
        
        Args:
            locale: 语言代码，如 'zh-cn', 'en-us'
            
        Returns:
            List: 分类列表
        """
        try:
            if locale:
                url = f"{self.base_url}/api/v2/help_center/{locale}/categories.json"
            else:
                url = f"{self.base_url}/api/v2/help_center/categories.json"

            all_categories = []
            page = 1

            while True:
                params = {'page': page, 'per_page': 100}
                self.logger.info(f"获取分类列表 - 页面 {page}")

                response = requests.get(url, auth=self.auth, headers=self.headers,
                                        params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    categories = data.get('categories', [])

                    if not categories:
                        break

                    all_categories.extend(categories)
                    self.logger.info(f"获取到 {len(categories)} 个分类")

                    # 检查是否还有更多页面
                    if len(categories) < 100:
                        break

                    page += 1
                    time.sleep(0.5)  # 避免API限流
                else:
                    self.logger.error(f"获取分类失败: {response.status_code} - {response.text}")
                    break

            self.logger.info(f"总共获取到 {len(all_categories)} 个分类")
            return all_categories

        except Exception as e:
            self.logger.exception(f"获取分类时出错: {e}")
            return []

    def get_sections(self, category_id: int = None, locale: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取章节列表
        
        Args:
            category_id: 分类ID，如果指定则只获取该分类下的章节
            locale: 语言代码
            
        Returns:
            List: 章节列表
        """
        try:
            if category_id:
                if locale:
                    url = f"{self.base_url}/api/v2/help_center/{locale}/categories/{category_id}/sections.json"
                else:
                    url = f"{self.base_url}/api/v2/help_center/categories/{category_id}/sections.json"
            else:
                if locale:
                    url = f"{self.base_url}/api/v2/help_center/{locale}/sections.json"
                else:
                    url = f"{self.base_url}/api/v2/help_center/sections.json"

            all_sections = []
            page = 1

            while True:
                params = {'page': page, 'per_page': 100}
                self.logger.debug(f"获取章节列表 - 页面 {page}")

                response = requests.get(url, auth=self.auth, headers=self.headers,
                                        params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    sections = data.get('sections', [])

                    if not sections:
                        break

                    all_sections.extend(sections)
                    self.logger.debug(f"获取到 {len(sections)} 个章节")

                    if len(sections) < 100:
                        break

                    page += 1
                    time.sleep(0.2)
                else:
                    self.logger.error(f"获取章节失败: {response.status_code} - {response.text}")
                    break

            return all_sections

        except Exception as e:
            self.logger.exception(f"获取章节时出错: {e}")
            return []

    def get_articles(self, section_id: int = None, locale: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取文档列表
        
        Args:
            section_id: 章节ID，如果指定则只获取该章节下的文档
            locale: 语言代码
            
        Returns:
            List: 文档列表
        """
        try:
            if section_id:
                if locale:
                    url = f"{self.base_url}/api/v2/help_center/{locale}/sections/{section_id}/articles.json"
                else:
                    url = f"{self.base_url}/api/v2/help_center/sections/{section_id}/articles.json"
            else:
                if locale:
                    url = f"{self.base_url}/api/v2/help_center/{locale}/articles.json"
                else:
                    url = f"{self.base_url}/api/v2/help_center/articles.json"

            all_articles = []
            page = 1

            while True:
                params = {'page': page, 'per_page': 100}
                self.logger.debug(f"获取文档列表 - 页面 {page}")

                response = requests.get(url, auth=self.auth, headers=self.headers,
                                        params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])

                    if not articles:
                        break

                    all_articles.extend(articles)
                    self.logger.debug(f"获取到 {len(articles)} 篇文档")

                    if len(articles) < 100:
                        break

                    page += 1
                    time.sleep(0.2)
                else:
                    self.logger.error(f"获取文档失败: {response.status_code} - {response.text}")
                    break

            return all_articles

        except Exception as e:
            self.logger.exception(f"获取文档时出错: {e}")
            return []

    def get_article_details(self, article_id: int, locale: Optional[str] = None) -> Dict[str, Any]:
        """
        获取文档详细信息
        
        Args:
            article_id: 文档ID
            locale: 语言代码
            
        Returns:
            Dict: 文档详细信息
        """
        try:
            if locale:
                url = f"{self.base_url}/api/v2/help_center/{locale}/articles/{article_id}.json"
            else:
                url = f"{self.base_url}/api/v2/help_center/articles/{article_id}.json"

            response = requests.get(url, auth=self.auth, headers=self.headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data.get('article', {})
            else:
                self.logger.warning(f"获取文档 {article_id} 详情失败: {response.status_code}")
                return {}

        except Exception as e:
            self.logger.exception(f"获取文档 {article_id} 详情时出错: {e}")
            return {}

    def sync_help_center_data(self, locales: List[str] = None) -> Dict[str, Any]:
        """
        同步帮助中心数据
        
        Args:
            locales: 要同步的语言列表，如 ['zh-cn', 'en-us']
            
        Returns:
            Dict: 同步结果统计
        """
        start_time = datetime.now()
        self.logger.info(f"开始Zendesk帮助中心数据同步")

        if not locales:
            locales = ['en-us']  # 默认英文

        stats = {
            'categories': 0,
            'sections': 0,
            'articles': 0,
            'errors': 0,
            'start_time': start_time.isoformat(),
            'end_time': None,
            'locales': locales
        }

        try:
            # 创建数据存储目录
            _date = datetime.now().strftime('%Y%m%d')
            data_dir = f"{self.base_json_data_dir}/{_date}"
            markdown_dir = f"{self.base_markdown_data_dir}/{_date}"
            try:
                if os.path.exists(data_dir) and os.listdir(data_dir):
                    self.logger.warning(f"目录 {data_dir} 已存在且不为空，跳过同步")
                    return stats
            except (OSError, FileNotFoundError):
                # 目录不存在或无权限访问，继续创建
                pass
            os.makedirs(data_dir, exist_ok=True)

            all_data = {}

            for locale in locales:
                self.logger.info(f"开始同步语言: {locale}")
                locale_data = {
                    'categories': [],
                    'sections': [],
                    'articles': []
                }

                # 1. 获取所有分类
                categories = self.get_categories(locale)
                locale_data['categories'] = categories
                stats['categories'] += len(categories)

                self.logger.info(f"语言 {locale}: 找到 {len(categories)} 个分类")

                # 2. 获取每个分类下的章节和文档
                for category in categories:
                    category_id = category['id']
                    category_name = category.get('name', f'category_{category_id}')

                    self.logger.info(f"处理分类: {category_name} (ID: {category_id})")

                    # 获取分类下的章节
                    sections = self.get_sections(category_id, locale)
                    locale_data['sections'].extend(sections)
                    stats['sections'] += len(sections)

                    self.logger.info(f"分类 {category_name}: 找到 {len(sections)} 个章节")

                    # 获取每个章节下的文档
                    for section in sections:
                        section_id = section['id']
                        section_name = section.get('name', f'section_{section_id}')

                        self.logger.info(f"处理章节: {section_name} (ID: {section_id})")

                        articles = self.get_articles(section_id, locale)

                        # 获取每篇文档的详细信息
                        for article in articles:
                            article_id = article['id']
                            article_details = self.get_article_details(article_id, locale)

                            if article_details:
                                locale_data['articles'].append(article_details)
                                stats['articles'] += 1

                                self.logger.debug(f"已处理文档: {article_details.get('title', '')} (ID: {article_id})")

                        # 避免请求过于频繁
                        time.sleep(0.1)

                # 保存该语言的所有数据
                all_data[locale] = locale_data

                # 保存到文件
                locale_file = os.path.join(data_dir, f"{locale}_help_center_data.json")
                with open(locale_file, 'w', encoding='utf-8') as f:
                    json.dump(locale_data, f, ensure_ascii=False, indent=2)

                self.logger.info(f"语言 {locale} 数据同步完成")

            # 保存汇总数据
            all_data_file = f"{data_dir}/all_help_center_data.json"
            with open(all_data_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)

            # 保存统计信息
            end_time = datetime.now()
            stats['end_time'] = end_time.isoformat()
            stats['duration_seconds'] = (end_time - start_time).total_seconds()

            with open(f"{data_dir}/sync_stats.json", 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            self.logger.info(f"帮助中心数据同步完成！统计: {stats}")
            self.logger.info(f"数据保存在: {data_dir}")

            # 默认开启：自动转换为Markdown文件
            self.logger.info("开始自动转换为Markdown文件...")


            convert_success = self.convert_to_markdown_files(all_data_file,markdown_dir)

            if convert_success:
                self.logger.info(f"Markdown文件转换成功！文件保存在: {markdown_dir}")
                stats['markdown_conversion'] = True
                stats['markdown_dir'] = markdown_dir
            else:
                self.logger.error("Markdown文件转换失败！")
                stats['markdown_conversion'] = False
                stats['errors'] += 1

            return stats

        except Exception as e:
            self.logger.exception(f"数据同步失败: {e}")
            stats['errors'] += 1
            stats['error_message'] = str(e)
            return stats

    def convert_to_markdown_files(self, json_file_path: str, output_dir: str):
        """
        将JSON文件中的文档转换为Markdown文件
        
        Args:
            json_file_path: JSON文件的完整路径
            output_dir: 输出目录路径
            
        Returns:
            bool: 转换是否成功
        """
        try:
            if os.path.exists(output_dir) and os.listdir(output_dir):
                self.logger.warning(f"目录 {output_dir} 已存在且不为空，跳过转换")
                return True
        except Exception:
            pass
        try:
            # 尝试导入html2text
            try:
                import html2text
            except ImportError:
                self.logger.error("需要安装html2text库: pip install html2text 或 uv add html2text")
                return False

            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 设置html2text转换器
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_emphasis = False
            h.body_width = 0  # 不限制行宽

            total_files = 0

            # 遍历所有语言
            for locale, locale_data in data.items():
                self.logger.info(f"处理语言: {locale}")

                # 创建语言子目录
                locale_dir = os.path.join(output_dir, locale)
                os.makedirs(locale_dir, exist_ok=True)

                articles = locale_data.get('articles', [])
                self.logger.info(f"找到 {len(articles)} 篇文档")

                # 处理每篇文档
                for article in articles:
                    name = article.get('name', article.get('title', f"article_{article.get('id', 'unknown')}"))
                    body = article.get('body', '')
                    article_id = article.get('id', 'unknown')

                    if not body:
                        self.logger.debug(f"跳过空文档: {name}")
                        continue

                    # 转换HTML为Markdown
                    try:
                        markdown_body = h.handle(body)
                        # 清理多余的空行
                        markdown_body = '\n'.join(
                            line for line in markdown_body.split('\n') if line.strip() or line == '')
                    except Exception as e:
                        self.logger.warning(f"转换HTML失败 {name}: {e}")
                        markdown_body = body  # 使用原始内容

                    # 创建Markdown内容，name作为一级标题
                    markdown_content = f"# {name}\n\n{markdown_body}"

                    # 创建安全的文件名
                    safe_filename = self._make_safe_filename(name, article_id)
                    file_path = os.path.join(locale_dir, f"{safe_filename}.md")

                    # 保存文件
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(markdown_content)

                        total_files += 1
                        self.logger.debug(f"已保存: {file_path}")

                    except Exception as e:
                        self.logger.error(f"保存文件失败 {safe_filename}: {e}")

            self.logger.info(f"转换完成！总共生成 {total_files} 个Markdown文件")
            self.logger.info(f"文件保存在: {output_dir}")

            return True

        except Exception as e:
            self.logger.exception(f"转换过程出错: {e}")
            return False

    def _make_safe_filename(self, name: str, article_id: int) -> str:
        """
        创建安全的文件名
        
        Args:
            name: 原始文件名
            article_id: 文章ID
            
        Returns:
            str: 安全的文件名
        """
        import re

        # 移除或替换不安全的字符
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe_name = re.sub(r'\s+', '_', safe_name)  # 替换空格为下划线
        safe_name = safe_name.strip('._')  # 移除开头和结尾的点和下划线

        # 限制长度
        if len(safe_name) > 100:
            safe_name = safe_name[:100]

        # 如果名称为空或只有特殊字符，使用文章ID
        if not safe_name or safe_name.isspace():
            safe_name = f"article_{article_id}"

        # 添加文章ID以确保唯一性
        safe_name = f"{article_id}_{safe_name}"

        return safe_name

    def run_scheduled_sync(self, locales: List[str] = None):
        """运行定时同步任务"""

        def job():
            self.logger.info("开始执行定时同步任务")
            result = self.sync_help_center_data(locales=locales)
            if result.get('errors', 0) > 0:
                self.logger.error(f"同步任务完成，但有错误: {result}")
            else:
                self.logger.info(f"同步任务成功完成: {result}")

        return job


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Zendesk帮助中心文档同步工具')
    parser.add_argument('--subdomain', required=True, help='Zendesk子域名')
    parser.add_argument('--email', required=True, help='Zendesk用户邮箱')
    parser.add_argument('--token', required=True, help='Zendesk API Token')
    parser.add_argument('--locale', action='append', help='要同步的语言代码，可重复使用指定多个语言')
    parser.add_argument('--mode', choices=['once', 'schedule'], default='once',
                        help='运行模式: once(单次执行)、schedule(定时执行)')
    parser.add_argument('--interval', type=int, default=24,
                        help='定时执行的间隔小时数 (默认: 24)')

    args = parser.parse_args()

    try:

        # 其他模式需要API认证参数
        if not all([args.subdomain, args.email, args.token]):
            print("同步模式需要指定 --subdomain、--email 和 --token 参数")
            sys.exit(1)

        # 默认语言
        locales = args.locale if args.locale else ['en-us']

        syncer = ZendeskHelpCenterSync(args.subdomain, args.email, args.token)

        if args.mode == 'once':
            # 单次执行
            print(f"开始单次同步，语言: {', '.join(locales)}")
            result = syncer.sync_help_center_data(locales=locales)
            print(f"同步完成: {result}")

        elif args.mode == 'schedule':
            # 定时执行
            print(f"启动定时同步，每 {args.interval} 小时执行一次，语言: {', '.join(locales)}")
            job = syncer.run_scheduled_sync(locales=locales)

            # 设置定时任务
            schedule.every(args.interval).hours.do(job)

            # 立即执行一次
            job()

            # 保持运行
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次


    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
