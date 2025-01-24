import asyncio
import platform
from typing import Any
import fire
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team

class SpeakAloud(Action):
    """动作：在辩论中大声说话（争吵）"""
    PROMPT_TEMPLATE: str = """
    ## BACKGROUND
    Suppose you are {name}, you are in a debate with {opponent_name}.
    ## DEBATE HISTORY
    Previous rounds:
    {context}
    ## YOUR TURN
    Now it's your turn, you should closely respond to your opponent's latest argument, state your position, defend your arguments, and attack your opponent's arguments,
    craft a strong and emotional response in 80 words, in {name}'s rhetoric and viewpoints, your will argue:
    """
    name: str = "SpeakAloud"
    async def run(self, context: str, name: str, opponent_name: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context, name=name, opponent_name=opponent_name)
        rsp = await self._aask(prompt)
        return rsp
    
class Debator(Role):
    name: str = ""
    profile: str = ""
    opponent_name: str = ""
    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([SpeakAloud])
        self._watch([UserRequirement, SpeakAloud])
    async def _observe(self) -> int:
        await super()._observe()
        # accept messages sent (from opponent) to self, disregard own messages from the last round
        self.rc.news = [msg for msg in self.rc.news if msg.send_to == {self.name}]
        return len(self.rc.news)
    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo # 一个 SpeakAloud 的实例
        memories = self.get_memories()
        context = "\n".join(f"{msg.sent_from}: {msg.content}" for msg in memories)
        rsp = await todo.run(context=context, name=self.name, opponent_name=self.opponent_name)
        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.opponent_name,
        )
        self.rc.memory.add(msg)
        return msg
async def debate(idea: str, investment: float = 3.0, n_round: int = 5):
        """运行拜登-特朗普辩论，观看他们之间的友好对话 :) """
        Biden = Debator(name="Biden", profile="Democrat", opponent_name="Trump")
        Trump = Debator(name="Trump", profile="Republican", opponent_name="Biden")
        team = Team()
        team.hire([Biden, Trump])
        team.invest(investment)
        team.run_project(idea, send_to="Biden")  # 将辩论主题发送给拜登，让他先说话
        await team.run(n_round=n_round)

import typer
app = typer.Typer()

@app.command()
def main():
    idea = "Economic Policy: Discuss strategies and plans related to taxation, employment, fiscal budgeting, and economic growth."
    investment = 3.0
    n_round = 5
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(debate(idea, investment, n_round))

if __name__ == '__main__':
    app()