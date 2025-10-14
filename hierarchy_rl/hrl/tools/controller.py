#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/02/12
@Author  : tanghaoming
@File    : device_controller.py
@Desc    : Device control utility class for operating Android and PC devices
"""

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pyautogui
import pyperclip
import uiautomator2 as u2

class BaseController:
    """Base device controller class

    Provides common functionality for Android and PC controllers.
    """

    def get_screenshot(self, filepath: str = "./screenshot/screenshot.jpg") -> None:
        """Take a screenshot

        Args:
            filepath: Path to save the screenshot
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            self._take_screenshot(filepath)
            logger.info(f"Screenshot saved to: {filepath}")
        except Exception as e:
            logger.error(f"Screenshot failed: {str(e)}")

    def _take_screenshot(self, filepath: str) -> None:
        """Implementation method for taking screenshots, to be implemented by subclasses"""
        raise NotImplementedError

    def run_action(self, action: str) -> None:
        """Execute action

        Args:
            action: Action description string
        """
        logger.info(f"Executing action: {action}")
        # Use list to maintain action order
        action_handlers = [
            ("Run", lambda x: hasattr(self, "_handle_run") and self._handle_run(x)),
            ("Tell", lambda x: hasattr(self, "_handle_tell") and self._handle_tell(x)),
        ]

        for action_type, handler in action_handlers:
            if action_type in action:
                handler(action)
                break

    def _handle_tell(self, action: str) -> None:
        """Handle 'Tell' action"""
        # Get text from action
        text = self._extract_code(action)
        logger.info(f"Handling 'Tell' action: {text}")

    def _extract_code(self, action: str) -> str:
        """Extract code from action string

        Args:
            action: Action string

        Returns:
            str: Extracted code
        """
        start = action.find("(")
        end = action.rfind(")")
        if start != -1 and end != -1 and end > start:
            code = action[start + 1 : end]
            return code.strip("```").replace("\n", "; ")
        return ""

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Check if text contains Chinese characters

        Args:
            text: Text to check

        Returns:
            bool: Whether text contains Chinese characters
        """
        return any("\u4e00" <= char <= "\u9fff" for char in text)


class ControllerTool(BaseController):
    """Android device controller class

    Provides basic operations for Android devices, including clicking, swiping, input, etc.
    """

    def __init__(self):
        """Initialize Android controller"""
        try:
            self.device = u2.connect()  # Connect device
            u2.enable_pretty_logging()
            self.device.set_input_ime(False)  # Switch input method
        except Exception as e:
            logger.error(f"Failed to initialize Android controller: {str(e)}")
            raise

    def _take_screenshot(self, filepath: str) -> None:
        """进行截图，并保存到file_path, Implement screenshot function for Android device"""
        self.device.screenshot(filepath)

    def get_screen_xml(self, location_info: str = "center") -> List[Dict]:
        """Get screen XML information

        Args:
            location_info: Location information format ('center' or 'bbox')

        Returns:
            List[Dict]: List containing element information
        """
        result = []
        screen_height = self.device.window_size()[1]
        xml = self.device.dump_hierarchy()
        root = ET.fromstring(xml)

        def get_element_text(element: ET.Element) -> str:
            """Recursively get element text"""
            if element.attrib.get("text"):
                return element.attrib.get("text")
            for child in element:
                text = get_element_text(child)
                if text:
                    return text
            return ""

        for elem in root.iter():
            elem_class = elem.attrib.get("class", "")
            clickable = elem.attrib.get("clickable", "false")
            focusable = elem.attrib.get("focusable", "false")
            elem_text = get_element_text(elem)
            elem_id = elem.attrib.get("resource-id", "")
            elem_desc = elem.attrib.get("content-desc", "")

            bounds = elem.attrib.get("bounds", "")
            if bounds:
                bounds = bounds.replace("][", ",").replace("[", "").replace("]", "")
                bounds = list(map(int, bounds.split(",")))

                if bounds and (bounds[3] - bounds[1]) > screen_height / 2:
                    continue

                if clickable == "true" or (
                    focusable == "true" and (elem_class == "android.widget.EditText" or elem_class == "android.widget.TextView")
                ):
                    center_x = int((bounds[0] + bounds[2]) / 2)
                    center_y = int((bounds[1] + bounds[3]) / 2)

                    result.append(
                        {
                            "coordinates": [center_x, center_y] if location_info == "center" else bounds,
                            "text": f"Class={elem_class}, Text={elem_text}, ID={elem_id}, Content-desc={elem_desc}, Bounds={bounds}",
                        }
                    )

        return result

    def get_all_packages(self) -> List[str]:
        """Get all installed app package names

        Returns:
            List[str]: List of package names
        """
        return self.device.app_list()

    def get_current_app_package(self) -> str:
        """Get current running app's package name

        Returns:
            str: Current app package name
        """
        return self.device.app_current()["package"]

    def open_app(self, package_name: str) -> bool:
        """Launch application

        Args:
            package_name: Application package name

        Returns:
            bool: Whether launch was successful
        """
        package_name = package_name.split(":")[-1].strip()
        try:
            installed_packages = self.get_all_packages()
            if package_name not in installed_packages:
                logger.error(f"App {package_name} is not installed")
                return False

            self.device.app_start(package_name)
            logger.info(f"Successfully launched app: {package_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to launch app: {str(e)}")
            return False

    def _handle_run(self, action: str) -> None:
        """Handle 'Run' action"""
        code = self._extract_code(action)
        code = code.replace("self.device.tap(", "self.device.click(")
        code = self._add_ime_control(code)
        logger.info(f"Executing code: {code}")
        exec(code)

    def _add_ime_control(self, code: str) -> str:
        """Add input method control to code

        Args:
            code: Original code

        Returns:
            str: Code with input method control added
        """
        matches = re.finditer(r'self\.device\.send_keys\("""(.*?)"""(?:, clear=True)?\);', code)
        modified_code = code
        offset = 0

        for match in matches:
            send_keys = match.group(0)
            new_send_keys = f"self.device.set_input_ime(True); time.sleep(0.5); {send_keys} time.sleep(0.5); self.device.set_input_ime(False);"

            start_index = match.start() + offset
            end_index = match.end() + offset

            modified_code = modified_code[:start_index] + new_send_keys + modified_code[end_index:]
            offset += len(new_send_keys) - len(send_keys)

        return modified_code
