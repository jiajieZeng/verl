#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc    : Executor 类，作为 HRL 框架的环境交互层。
"""
import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

# 导入项目自身的工具和 prompts
from ..tools.controller import ControllerTool
from ..prompts.executor_prompt import Android_prompt  # 复用 OSAgent 中的 prompt 逻辑

# 可选的工具导入，如果某些依赖库（如 OCR）未安装，则优雅地回退，避免程序崩溃
try:
    from ..tools.ocr import OCRTool  # type: ignore
except Exception:
    OCRTool = None  # type: ignore
try:
    from ..tools.icon import IconDetectTool  # type: ignore
except Exception:
    IconDetectTool = None  # type: ignore


@dataclass
class StepResult:
    """
    用于表示 Executor 在环境中执行单步操作后结果的数据类。
    这个结构由 `step` 方法返回。
    """
    # 执行动作后的环境新状态。
    # 它由感知到的 UI 元素列表和屏幕尺寸组成。
    next_state: Tuple[List[Dict[str, Any]], int, int]
    # 执行该动作获得的即时奖励。
    # 在这个 HRL 设置中，Executor 的奖励是在训练器循环中计算的，所以这里通常是一个占位符（例如 0.0）。
    reward: float
    # 一个布尔标志，指示一个 episode (回合) 是否已经终止。
    # 终止的原因可能是出错、任务完成或达到某个终结状态。
    done: bool
    # 一个字典，用于存放关于该步骤的辅助信息，
    # 对调试、日志记录以及在训练器中计算奖励很有用。
    info: Dict[str, Any] = field(default_factory=dict)


class Executor:
    """
    Executor (执行器) 扮演代理的“手”和“眼”，直接与操作系统环境进行交互。
    它封装了感知屏幕（perception）和执行命令（action）的逻辑。
    """

    def __init__(
        self,
        actor_wg: RayWorkerGroup,
        tokenizer: Any,
        saving_folder: str,
        platform: str = "Android",
        use_ocr: bool = True,
        quad_split_ocr: bool = False,
        use_icon_detect: bool = True,
        use_icon_caption: bool = True,
        extend_xml_infos: bool = True,
        location_info: str = "center",
        draw_text_box: bool = False,
    ) -> None:
        """初始化 Executor 及其所有工具。"""
        self.actor_wg = actor_wg
        self.tokenizer = tokenizer
        self.platform = platform
        self.use_ocr = use_ocr
        self.use_icon_detect = use_icon_detect
        self.use_icon_caption = use_icon_caption
        self.quad_split_ocr = quad_split_ocr
        self.location_info = location_info
        self.extend_xml_infos = extend_xml_infos
        self.knowledge_base_path = knowledge_base_path
        self.draw_text_box = draw_text_box
        
        self.controller = ControllerTool()
        self.ocr_tool = OCRTool() if (self.use_ocr and OCRTool) else None
        self.icon_tool = IconDetectTool(self.actor_wg, self.tokenizer) if (self.use_icon_detect and IconDetectTool) else None
        self.prompt_utils = Android_prompt()
        self.saving_folder = saving_folder
        self.detailed_saving_foler = None

        self.app_info = None
        self.thought: str = ""                                  # Current thinking content
        self.thought_history: List[str] = []                    # Historical thinking records list
        self.summary_history: List[str] = []                    # Historical operation summary list
        self.action_history: List[str] = []                     # Historical executed action list
        self.summary: str = ""                                  # Current operation summary
        self.action: str = ""                                   # Current executed action
        self.error_flag: bool = False                           # Error flag
        self.perception_infos: List[Dict] = []                  # Current perception information list
        self.last_perception_infos: List[Dict] = []             # Previous perception information list
        self.width: int = 0                                     # Screen width
        self.height: int = 0                                    # Screen height

        self.initial_screenshot_path: str = None
        self.initial_screenshot_som_path: str = None
        self.screenshot_path: str = None                         # Current screenshot path
        self.screenshot_som_path: str = None                     # Current screenshot with SOM path
        self.global_step: int = 0                                # Global step count
        self.episode_step: int = 0                               # Step count within the current episode
        self.action_step: int = 0                                # Action step count within the current episode

    def set_step_num(self, global_step: int, episode_step: int, action_step: int) -> None:
        self.global_step = global_step
        self.episode_step = episode_step
        self.action_step = action_step

    def reset(self, app_info: Any, global_step: int, episode_idx: int) -> Tuple[List[Dict[str, Any]], int, int]:
        """重置环境到初始状态并返回初始观察。"""
        PrettyPrinter.status("【Executor】: Resetting environment...")

        # 3. Reset all internal states
        self.thought = ""
        self.thought_history = []
        self.summary_history = []
        self.action_history = []
        self.summary = ""
        self.action = ""
        self.error_flag = False
        self.perception_infos = []
        self.last_perception_infos = []
        self.width = 0
        self.height = 0
        self.initial_screenshot_path = ""

        self.global_step = global_step
        self.episode_step = episode_idx
        
        # 1. Set app info first
        self.app_info = app_info
        self.detailed_saving_foler = file.path.join(self.saving_folder, app_info.name, str(global_step), str(episode_idx))
        os.makedirs(self.detailed_saving_foler, exist_ok=True)  
        self.initial_screenshot_path = file.path.join(self.detailed_saving_foler, "initial_screenshot.jpg")
        self.initial_screenshot_som_path = file.path.join(self.detailed_saving_foler, "initial_screenshot_som.png")
        # 2. Open the app
        if self.platform == "Android" and self.app_info:
            package_name = getattr(self.app_info, 'package_name', None) or getattr(self.app_info, 'app_path', None)
            if package_name:
                PrettyPrinter.status(f"【Executor】: Opening specified app: {package_name}")
                self.controller.open_app(package_name)
                time.sleep(5)  # Wait for app to launch
        


        # 4. Get and return the initial state
        # 打开app后要获取初始状态，包含开始的perception info, width, height, screenshot, screenshot_som
        if self.app_info:
            perception_infos, width, height = asyncio.run(self.get_state())
        return perception_infos, width, height, self.initial_screenshot_path, self.initial_screenshot_som_path


    def pack_screenshot_path(self, use_som=False) -> None:
        """Pack screenshot path with step info"""
        if use_som:
            self.screenshot_som_path = file.path.join(self.detailed_saving_foler, f"screenshot_{self.global_step}_{self.episode_step}_{self.action_step}_som.png")
        self.screenshot_path = file.path.join(self.detailed_saving_foler, f"screenshot_{self.global_step}_{self.episode_step}_{self.action_step}.jpg")

    async def get_state(self) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        感知当前屏幕状态，提取详细 UI 元素信息（整合了 OCR、图标检测与描述等）。
        This is the single source of truth for environment perception.
        """
        # 1. Get screenshot and dimensions
        self.pack_screenshot_path(use_som=False)
        self.pack_screenshot_path(use_som=True)
        self.controller.get_screenshot(self.screenshot_path)
        image = Image.open(self.screenshot_path)
        width, height = image.size
        
        perception_infos: List[Dict[str, Any]] = []
        mark_number = 0

        # 2. OCR Processing
        text, text_coordinates = [], []
        if self.use_ocr:
            text, text_coordinates = self.ocr_tool.ocr(self.screenshot_path, split=self.quad_split_ocr)

        # 3. Icon Detection
        icon_coordinates = []
        if self.use_icon_detect:
            icon_coordinates = self.icon_tool.detect(self.screenshot_path)

        # 这是干嘛的？
        output_image_path = self.screenshot_path
        if self.use_ocr and self.use_icon_detect and self.draw_text_box:
            rec_list = text_coordinates + icon_coordinates
            self._draw_bounding_boxes(self.screenshot_file, copy.deepcopy(rec_list), self.screenshot_som_file, self.font_path)
        elif self.use_icon_detect:
            self._draw_bounding_boxes(
                self.screenshot_file, copy.deepcopy(icon_coordinates), self.screenshot_som_file, self.font_path
            )
        else:
            output_image_path = self.screenshot_file

        # build perception information
        # Add ORC text Information
        if self.use_ocr:
            for i in range(len(text_coordinates)):
                mark_number += 1
                if self.use_som and self.draw_text_box:
                    perception_info = {
                        "text": f"mark number: {mark_number} text: {text[i]}",
                        "bbox": text_coordinates[i],
                    }
                else:
                    perception_info = {"text": f"text: {text[i]}", "bbox": text_coordinates[i]}
                perception_infos.append(perception_info)

        # Add icon information
        if self.use_icon_detect:
            for i in range(len(icon_coordinates)):
                mark_number += 1
                if self.use_som:
                    perception_info = {"text": f"mark number: {mark_number} icon", "bbox": icon_coordinates[i]}
                else:
                    perception_info = {"text": "icon", "bbox": icon_coordinates[i]}
                perception_infos.append(perception_info)
        
        # 4. Icon Description
        if self.use_icon_detect and self.use_icon_caption:
            icon_indices = [i for i in range(len(perception_infos)) if "icon" in perception_infos[i]["text"]]
            if icon_indices:
                icon_boxes = [perception_infos[i]["bbox"] for i in icon_indices]
                descriptions = await self.icon_tool.caption(self.screenshot_path, icon_boxes, platform=self.platform)

                # Add description to perception information
                for idx, desc_idx in enumerate(icon_indices):
                    if descriptions.get(idx + 1):
                        perception_infos[desc_idx]["text"] += ": " + descriptions[idx + 1].replace("\n", " ")

        # 5. According to parameter modify coordinate information
        if self.location_info == "center":
            for i in range(len(perception_infos)):
                x1, y1, x2, y2 = perception_infos[i]["bbox"]
                perception_infos[i]["bbox"] = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        elif self.location_info == "icon_center":
            for i in range(len(perception_infos)):
                if "icon" in perception_infos[i]["text"]:
                    x1, y1, x2, y2 = perception_infos[i]["bbox"]
                    perception_infos[i]["bbox"] = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        
        # IF exend_xml_infos is enabled, then get XML information
        if self.extend_xml_infos and self.platform in ["Android"]:
            xml_results = self.controller.get_screen_xml(self.location_info)
            perception_infos.extend(xml_results)

        # 8. Update instance state and return
        self.width, self.height = width, height
        self.perception_infos = perception_infos
        if self.action_step == 0:
            self.initial_screenshot_path = self.screenshot_path
            self.initial_screenshot_som_path = self.screenshot_som_path
        return perception_infos, width, height

    async def step(self, action: str) -> StepResult:
        """在环境中执行单个动作，并返回结果。"""
        done, info = False, {"action_ok": False, "error_message": ""}
        self.last_perception_infos = self.perception_infos.copy()

        try:
            if "Stop" in action:
                PrettyPrinter.status("【Executor】: 'Stop' action received. Episode terminated.")
                done, info["action_ok"] = True, True
            else:
                self.controller.run_action(action)
                info["action_ok"] = True
        except Exception as e:
            PrettyPrinter.status(f"【Executor】: Action failed! Error: {e}")
            done, info["error_message"] = True, str(e)
        
        time.sleep(1.0)
        
        next_state = await self.get_state()
        reward = 0.0

        self.action = action
        self.action_history.append(action)
        return StepResult(next_state=next_state, reward=reward, done=done, info=info)


    def get_context(self) -> Dict[str, Any]:
        """收集并返回应用的宏观上下文信息，供 Planner 使用。"""
        if not self.app_info: return {}
        return {
            "app_name": self.app_info.name,
            "package_name": self.app_info.package_name,
            "initial_screenshot_path": self.initial_screenshot_path,
            "manifest_summary": self._get_manifest_info_from_apk(self.app_info.path),
            "coverage": [], # TODO: Implement real coverage tracking
            "found_defects": [], # TODO: Implement real defect tracking
            "plan_history": self.summary_history,
        }

    def _draw_bounding_boxes(
        self, image_path: str, coordinates: List[List[int]], output_path: str, font_path: str
    ) -> None:
        """Draw numbered coordinate boxes on the image.

        Args:
            image_path (str): Image path.
            coordinates (list): List of coordinate boxes, each box is a list of four elements [x1, y1, x2, y2].
            output_path (str): Output image path.
            font_path (str): Font path.
        """
        # Open image and get dimensions
        image = Image.open(image_path)
        height = image.size[1]

        # Calculate drawing parameters
        line_width = int(height * 0.0025)
        font_size = int(height * 0.012)
        text_offset_x = line_width
        text_offset_y = int(height * 0.013)

        # Generate random colors for each bounding box
        colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(coordinates))]

        # Draw bounding boxes and numbers
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)

        for i, (coord, color) in enumerate(zip(coordinates, colors)):
            # Draw bounding box using RGB color directly
            draw.rectangle(coord, outline=color, width=line_width)

            # Calculate text position and draw number
            text_x = coord[0] + text_offset_x
            text_y = max(0, coord[1] - text_offset_y)
            draw.text((text_x, text_y), str(i + 1), fill=color, font=font)

        # Save result
        image.convert("RGB").save(output_path)

    def _save_iteration_images(self, iter_num: int) -> None:
        """Save original and annotated images for current iteration.

        Args:
            iter_num: Current iteration number
        """
        # Build file paths TODO: 路径待定, 留个空缺提示符
        origin_path = f"{self.save_img}/origin_{iter_num}.jpg"
        draw_path = f"{self.save_img}/draw_{iter_num}.jpg"

        # Copy image files
        shutil.copy2(self.screenshot_file, origin_path)
        shutil.copy2(self.output_image_path, draw_path)

    def _update_screenshot_files(self) -> None:
        """Update screenshot files"""
        # Update normal screenshot
        last_screenshot = Path(self.last_screenshot_file)
        if last_screenshot.exists():
            last_screenshot.unlink()
        Path(self.screenshot_file).rename(last_screenshot)

        # Update SOM screenshot
        if self.use_som:
            last_screenshot_som = Path(self.last_screenshot_som_file)
            if last_screenshot_som.exists():
                last_screenshot_som.unlink()
            Path(self.screenshot_som_file).rename(last_screenshot_som)

    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get action history, including thoughts, summaries, actions, optional memories and reflections.
        Returns:
            list: A list of dictionaries, each dictionary represents a historical record of an action step.
                  Each dictionary contains "thought", "summary", "action",
                  and optional "memory" and "reflection".
        """
        outputs = []
        # Use zip to pair corresponding elements of three historical lists and use enumerate to get index
        for i, (thought, summary, action) in enumerate(
            zip(self.rc.thought_history, self.rc.summary_history, self.rc.action_history)
        ):
            output = {
                "thought": thought,
                "summary": summary,
                "action": action,
            }  # Current step thought  # Current step summary  # Current step action
            # If memory switch is enabled, add memory information
            if self.use_memory:
                output["memory"] = self.rc.memory[i]
            # If reflection switch is enabled, add reflection information
            if self.use_reflection:
                output["reflection"] = self.rc.reflection_thought_history[i]
            outputs.append(output)  # Add current step information to output list
        return outputs


    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Generate operation decision failed, {retry_state.attempt_number}th retry: {str(retry_state.outcome.exception())}"
        ),
        reraise=True,
    )
    async def _think(self) -> bool:
        """Generate operation decisions"""
        # Add preset knowledge
        add_info = self.add_info
        # Add application information to prompt
        if self.platform == "Android":
            info = self._get_app_info()
            if info:
                add_info += " ".join(info) if isinstance(info, list) else info
            else:
                info = "No add_info"
            logger.info(f"\n\n\n\n\n\n#### add_info:{info}\n\n")
        else:
            logger.info("Knowledge base currently only implemented for Android")

        # Generate action
        ctx = ActionPromptContext(
            instruction=self.instruction,
            clickable_infos=self.rc.perception_infos,
            width=self.width,
            height=self.height,
            thought_history=self.rc.thought_history,
            summary_history=self.rc.summary_history,
            action_history=self.rc.action_history,
            reflection_thought_history=self.rc.reflection_thought_history,
            last_summary=self.rc.summary,
            last_action=self.rc.action,
            reflection_thought=self.rc.reflection_thought,
            add_info=add_info,
            error_flag=self.rc.error_flag,
            completed_content=self.rc.completed_requirements,
            memory=self.rc.memory,
            task_list=self.rc.task_list,
            use_som=self.use_som,
            location_info=self.location_info,
        )

        prompt_action = self.prompt_utils.get_action_prompt(ctx)
        logger.info(
            f"\n\n######################## prompt_action:\n{prompt_action}\n\n######################## prompt_action end\n\n\n\n"
        )

        # Call LLM to generate decision
        images = [encode_image(self.screenshot_file)]
        if self.use_som:
            images.append(encode_image(self.screenshot_som_file))

        # Use custom system prompt or default prompt
        system_msg = (
            self.system_prompt
            if self.system_prompt
            else f"You are a helpful AI {'mobile phone' if self.platform=='Android' else 'PC'} operating assistant. You need to help me operate the device to complete the user's instruction."
        )

        output_action = await self.llm.aask(
            prompt_action,
            system_msgs=[system_msg],
            images=images,
            stream=False,
        )

        # Parse output
        self.rc.thought = (
            output_action.split("### Thought ###")[-1]
            .split("### Action ###")[0]
            .replace("\n", " ")
            .replace(":", "")
            .replace("  ", " ")
            .strip()
        )
        self.rc.action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].strip()
        self.rc.summary = (
            output_action.split("### Operation ###")[-1].split("### Task List ###")[0].strip().replace("\n", "\\n")
        )
        self.rc.task_list = output_action.split("### Task List ###")[-1].strip()

        logger.info(
            f"\n\n######################## output_action:\n{output_action}\n\n######################## output_action end\n\n\n\n"
        )

        if self.rc.action.startswith("Stop"):
            return False
        else:
            return True

    async def _act(self) -> Message:
        """Execute action step"""
        if self.use_chrome_debugger:
            # Store browser logs from before action execution in previous action log. Note: Need a log for step 0 here since mgx web testing is not started by osagent
            self.rc.webbrowser_console_logs.append(self.chrome_debugger.get_new_messages())

        self.run_action_failed = False
        self.run_action_failed_exception = ""

        # Execute action
        if "Stop" in self.rc.action:
            # If it's a stop operation, end the loop
            return AIMessage(content=self.rc.action, cause_by=Action)
        elif "Open App" in self.rc.action:
            await self._handle_open_app()
        else:
            # Execute other actions
            try:
                if self.platform in ["Android", "Windows"]:
                    self.controller.run_action(self.rc.action)
                else:
                    logger.error("Currently only supports Android and Windows")
            except Exception as e:
                # For direct exit when using tell in automg
                if isinstance(e, SystemExit) and e.code == 0:
                    return AIMessage(content=self.rc.action, cause_by=Action)
                logger.error(f"run action failed: {e}")
                self.run_action_failed = True
                self.run_action_failed_exception = e

        time.sleep(0.5)
        # Save previous perception information and screenshot
        self.rc.last_perception_infos = copy.deepcopy(self.rc.perception_infos)

        # Update screenshot files
        self._update_screenshot_files()

        # Get new perception information
        self.rc.perception_infos, self.width, self.height, self.output_image_path = await self._get_perception_infos(
            self.screenshot_file, self.screenshot_som_file
        )

        # Save images
        self._save_iteration_images(self.rc.iter)

        # Execute memory task asynchronously
        memory_task = None
        if self.use_memory:
            memory_task = asyncio.create_task(self._async_memory_task(self.instruction, self.screenshot_file))

        # Update history records
        self.rc.thought_history.append(self.rc.thought)
        self.rc.summary_history.append(self.rc.summary)
        self.rc.action_history.append(self.rc.action)

        if self.run_action_failed:
            self.rc.reflection_thought_history.append(
                f"ERROR(run action code filed): {self.run_action_failed_exception}\\n "
            )
            self.rc.error_flag = True

        elif self.use_reflection:
            # Execute reflection
            reflect, self.rc.reflection_thought = await self._reflection(
                self.instruction,
                self.rc.last_perception_infos,
                self.rc.perception_infos,
                self.width,
                self.height,
                self.rc.summary,
                self.rc.action,
                self.add_info,
                self.last_screenshot_file,
                self.screenshot_file,
            )
            self.rc.reflection_thought_history.append(self.rc.reflection_thought)
            if reflect == "CORRECT":
                self.rc.error_flag = False
            elif reflect == "ERROR":
                self.rc.error_flag = True

        # Clean up screenshots
        Path(self.last_screenshot_som_file if self.use_som else self.last_screenshot_file).unlink()

        # Wait for memory task to complete and save results
        if memory_task:
            memory_content = await memory_task
            self.rc.memory.append(memory_content)

        return AIMessage(content=self.rc.action, cause_by=Action)

    def _get_timestamped_paths(self) -> None:
        """Update file paths with timestamps"""
        current_time = time.strftime("%Y%m%d%H%M")

        # Base paths
        log_dir = Path(self.log_dirs) / current_time
        self.save_info = str(log_dir / "info.txt")
        self.save_img = str(log_dir)

        # Screenshot related paths
        self.screenshot_dir = log_dir / "screenshot"
        self.screenshot_file = str(self.screenshot_dir / "screenshot.jpg")
        self.screenshot_som_file = str(self.screenshot_dir / "screenshot_som.png")
        self.last_screenshot_file = str(self.screenshot_dir / "last_screenshot.jpg")
        self.last_screenshot_som_file = str(self.screenshot_dir / "last_screenshot_som.png")


    # async def get_state(self) -> Tuple[List[Dict[str, Any]], int, int]:
    #     """
    #     感知当前屏幕状态，提取详细 UI 元素信息（整合了 OCR、图标检测与描述等）。
    #     This is the single source of truth for environment perception.
    #     """
    #     # 1. Get screenshot and dimensions
    #     self.screenshot_path = self.pack_screenshot_path(use_som=False)
    #     self.controller.get_screenshot(self.screenshot_path)
    #     image = Image.open(self.screenshot_path)
    #     width, height = image.size
        
    #     perception_infos: List[Dict[str, Any]] = []
    #     mark_number = 0

    #     # 2. OCR Processing
    #     if self.use_ocr:
    #         ocr_texts, ocr_boxes = self.ocr_tool.ocr(self.screenshot_path, split=self.quad_split_ocr)
    #         for text, box in zip(ocr_texts, ocr_boxes):
    #             # mark_number += 1
    #             perception_infos.append({
    #                 "type": "text",
    #                 "text": f"mark number: {mark_number} text: {text}",
    #                 "bbox": box
    #             })

    #     # 3. Icon Detection
    #     icon_boxes = []
    #     if self.icon_tool:
    #         icon_boxes = self.icon_tool.detect(self.screenshot_path)
    #         for box in icon_boxes:
    #             mark_number += 1
    #             perception_infos.append({
    #                 "type": "icon",
    #                 "text": f"mark number: {mark_number} icon",
    #                 "bbox": box
    #             })

    #     # 4. Icon Captioning
    #     if self.icon_tool and self.use_icon_caption and icon_boxes:
    #         icon_indices = [i for i, info in enumerate(perception_infos) if info["type"] == "icon"]
    #         actual_icon_boxes = [perception_infos[i]["bbox"] for i in icon_indices]
    #         if actual_icon_boxes:
    #             descriptions = await self.icon_tool.caption(self.screenshot_path, actual_icon_boxes, platform=self.platform)
    #             for i, desc_idx in enumerate(icon_indices):
    #                 if descriptions and descriptions.get(i + 1):
    #                     perception_infos[desc_idx]["text"] += ": " + descriptions[i + 1].replace("\n", " ")

    #     # 5. Extend with XML Info
    #     if self.extend_xml_infos and self.platform == "Android":
    #         xml_results = self.controller.get_screen_xml(self.location_info)
    #         perception_infos.extend(xml_results)

    #     # 6. Calculate final coordinates for all elements
    #     for item in perception_infos:
    #         if "bbox" in item and "coordinates" not in item: # XML results might already have coordinates
    #             x1, y1, x2, y2 = item["bbox"]
    #             if self.location_info == "center":
    #                 item["coordinates"] = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
    #             else:
    #                 item["coordinates"] = item["bbox"]
        
    #     # 7. (Side-effect) Draw annotated image if enabled
    #     if self.use_som:
    #         all_boxes = [info["bbox"] for info in perception_infos if "bbox" in info]
    #         self._draw_bounding_boxes(self.screenshot_path, all_boxes, self.output_image_path, self.font_path)

    #     # 8. Update instance state and return
    #     self.width, self.height = width, height
    #     self.perception_infos = perception_infos
    #     return perception_infos, width, height