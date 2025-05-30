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
            data_dir = f"data/zendesk_help_center/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            with open(f"{data_dir}/all_help_center_data.json", 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            # 保存统计信息
            end_time = datetime.now()
            stats['end_time'] = end_time.isoformat()
            stats['duration_seconds'] = (end_time - start_time).total_seconds()
            
            with open(f"{data_dir}/sync_stats.json", 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"帮助中心数据同步完成！统计: {stats}")
            self.logger.info(f"数据保存在: {data_dir}")
            
            return stats
            
        except Exception as e:
            self.logger.exception(f"数据同步失败: {e}")
            stats['errors'] += 1
            stats['error_message'] = str(e)
            return stats
    
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
                       help='运行模式: once(单次执行) 或 schedule(定时执行)')
    parser.add_argument('--interval', type=int, default=24,
                       help='定时执行的间隔小时数 (默认: 24)')
    
    args = parser.parse_args()
    
    # 默认语言
    locales = args.locale if args.locale else ['en-us']
    
    try:
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