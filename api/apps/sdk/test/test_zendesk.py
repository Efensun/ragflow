#!/usr/bin/env python3
"""
Zendesk integration test script
"""

import sys
import os

import requests

# 修正：向上4级目录找到项目根目录
# 当前: /path/to/project/api/apps/sdk/test/test_zendesk.py
# 需要: /path/to/project/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)
from api import settings
settings.init_settings()
from api.apps.sdk.zendesk import (
    list_switchboards,
    list_switchboard_integrations,
    get_zendesk_config,
    test_auth, get_auth_headers
)


def main():
    """测试Zendesk集成功能"""

    # 获取配置
    config = get_zendesk_config()
    app_id = config.get('app_id', 'your_app_id')

    if app_id == 'your_app_id':
        print("❌ Zendesk配置未设置，请先配置ZENDESK_CONFIG")
        return

    print(f"🔍 Testing Zendesk integration with App ID: {app_id}")

    # 测试1: 查询switchboards
    print("\n1. 查询Switchboards...")
    switchboards_result = list_switchboards(app_id)

    if "error" in switchboards_result:
        print(f"❌ 查询switchboards失败: {switchboards_result['error']}")
        return

    switchboards = switchboards_result.get('switchboards', [])
    print(f"✅ 找到 {len(switchboards)} 个switchboard(s)")

    if not switchboards:
        print("❌ 没有找到switchboard")
        return

    # 使用第一个switchboard
    switchboard = switchboards[0]
    switchboard_id = switchboard['id']
    print(f"📋 使用Switchboard: {switchboard_id}")

    # 测试2: 查询integrations
    print(f"\n2. 查询Switchboard Integrations...")
    integrations_result = list_switchboard_integrations(app_id, switchboard_id)

    if "error" in integrations_result:
        print(f"❌ 查询integrations失败: {integrations_result['error']}")
        return

    integrations = integrations_result.get('switchboardIntegrations', [])
    print(f"✅ 找到 {len(integrations)} 个integration(s)")

    # 显示integration详情
    for i, integration in enumerate(integrations, 1):
        print(f"  {i}. {integration.get('name')} ({integration.get('integrationType')})")
        print(f"     ID: {integration.get('id')}")
        print(f"     Integration ID: {integration.get('integrationId')}")
        print(f"     Next: {integration.get('nextSwitchboardIntegrationId')}")
        print()


def add_webhook_to_switchboard():
    """将您的webhook integration添加到switchboard"""
    try:
        headers = get_auth_headers()
        zendesk_config = get_zendesk_config()
        base_url = zendesk_config.get('base_url', 'https://api.smooch.io')
        app_id = "682d78bb5221c34405e9bf87"
        switchboard_id = "682d78bc5221c34405e9bf9d"

        url = f"{base_url}/v2/apps/{app_id}/switchboards/{switchboard_id}/switchboardIntegrations"

        payload = {
            "name": "test-customer-bot",
            "integrationId": "6838490e5cacb40e733fea1d",
            "deliverStandbyEvents": False,
            "nextSwitchboardIntegrationId": "682d78bc5221c34405e9bfae"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print("Response:", response.json())

    except Exception as e:
        print(f"Error adding webhook to switchboard: {e}")
        return {"error": str(e)}


def update_default_integration(app_id: str, switchboard_id: str, new_default_integration_id: str) -> dict:
    """
    更新switchboard的默认integration

    Args:
        app_id: Zendesk App ID
        switchboard_id: Switchboard ID
        new_default_integration_id: 新的默认integration ID

    Returns:
        dict: 更新结果
    """
    try:
        headers = get_auth_headers()
        zendesk_config = get_zendesk_config()
        base_url = zendesk_config.get('base_url', 'https://api.smooch.io')

        url = f"{base_url}/v2/apps/{app_id}/switchboards/{switchboard_id}"

        payload = {
            "defaultSwitchboardIntegrationId": new_default_integration_id
        }

        response = requests.patch(url, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print(f"Successfully updated default integration to: {new_default_integration_id}")
            return result
        else:
            print(f"Failed to update default integration: {response.status_code} - {response.text}")
            return {"error": f"Update failed: {response.status_code}"}

    except Exception as e:
        print(f"Error updating default integration: {e}")
        print ("error",str(e))



if __name__ == '__main__':
    update_default_integration("682d78bb5221c34405e9bf87","682d78bc5221c34405e9bf9d","683eb89c6cb3f4630b9e2c41")
    main()