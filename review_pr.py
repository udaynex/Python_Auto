import os
import github
from github import Github
import subprocess
import tempfile
import logging
import re
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_env_var(var_name, var_value):
    if not var_value:
        logger.error(f"{var_name} not found in environment variables. Please set it in .env file.")
        sys.exit(1)
    return var_value

# Updated to use GITHUB_TOKEN in CI, SEC_TOKEN locally
if os.getenv("GITHUB_ACTIONS") == "true":
    SEC_TOKEN = validate_env_var("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN"))
else:
    SEC_TOKEN = validate_env_var("SEC_TOKEN", os.getenv("SEC_TOKEN"))

GOOGLE_API_KEY = validate_env_var("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
REPO_NAME = os.getenv("REPO_NAME", "udaynex/SmallPythonProject")
try:
    PR_NUMBER = int(os.getenv("PR_NUMBER", "1"))
except ValueError:
    logger.error("PR_NUMBER must be a valid integer.")
    sys.exit(1)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Log environment variables (partial for security)
logger.info(f"Loaded token: {SEC_TOKEN[:4]}...")
logger.info(f"Loaded GOOGLE_API_KEY: {GOOGLE_API_KEY[:4]}...")
logger.info(f"Loaded REPO_NAME: {REPO_NAME}")
logger.info(f"Loaded PR_NUMBER: {PR_NUMBER}")
logger.info(f"Loaded GEMINI_MODEL: {GEMINI_MODEL}")

# Interactive prompt for local runs only
if os.getenv("GITHUB_ACTIONS") != "true":
    try:
        import inquirer
        questions = [
            inquirer.Text("repo_name", message="Enter repository name", default=REPO_NAME),
            inquirer.Text("pr_number", message="Enter PR number", default=str(PR_NUMBER), validate=lambda _, x: x.isdigit())
        ]
        answers = inquirer.prompt(questions)
        if not answers:
            logger.error("No input provided for repository or PR number. Exiting.")
            sys.exit(1)
        REPO_NAME = answers["repo_name"]
        PR_NUMBER = int(answers["pr_number"])
    except ImportError:
        logger.warning("inquirer not available, using environment variables")

# Initialize GitHub client
try:
    github_client = Github(SEC_TOKEN)
    repo = github_client.get_repo(REPO_NAME)
except github.GithubException as e:
    logger.error(f"Failed to initialize GitHub client: {str(e)}")
    sys.exit(1)

# Initialize LLM (Gemini AI)
try:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )
except Exception as e:
    logger.error(f"Failed to initialize Gemini AI LLM: {str(e)}")
    sys.exit(1)

# Define CrewAI Agents
code_reviewer = Agent(
    role="Code Reviewer",
    goal="Analyze Python code in a GitHub PR, identify issues using flake8, and generate clear review comments.",
    backstory="You are an experienced Python developer with expertise in code quality and best practices.",
    tools=[],
    llm=llm,
    verbose=True
)

code_fixer = Agent(
    role="Code Fixer",
    goal="Suggest and apply fixes for identified Python issues, creating a new PR with the changes.",
    backstory="You are a skilled developer who automates fixes for common Python issues.",
    tools=[],
    llm=llm,
    verbose=True
)

def create_flake8_config():
    """Create a basic flake8 configuration for the project."""
    flake8_config = """
    [flake8]
    max-line-length = 88
    select = E,F,W,B,B950
    ignore = E501
    """
    try:
        with open('.flake8', 'w') as f:
            f.write(flake8_config)
        logger.info("Created flake8 configuration")
    except Exception as e:
        logger.error(f"Failed to create flake8 config: {str(e)}")

def run_linter(file_content, file_path):
    """Run flake8 on the provided file content and return issues."""
    try:
        # Ensure flake8 config exists
        if not os.path.exists('.flake8'):
            create_flake8_config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        result = subprocess.run(["flake8", temp_file_path, "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s"], capture_output=True, text=True)
        os.unlink(temp_file_path)

        if result.returncode != 0 and not result.stdout:
            logger.warning(f"flake8 returned no output for {file_path}: {result.stderr}")
            return []

        if not result.stdout.strip():
            return []

        issues = result.stdout.splitlines()
        return issues
    except Exception as e:
        logger.error(f"Error running flake8 on {file_path}: {str(e)}")
        return []

def get_pr_files(pr_number):
    """Fetch Python files from a GitHub PR."""
    try:
        pr = repo.get_pull(pr_number)
        files = pr.get_files()
        py_files = [f for f in files if f.filename.endswith('.py')]
        if not py_files:
            logger.info("No Python files found in PR.")
            return []
        return py_files
    except github.GithubException as e:
        logger.error(f"Error fetching PR #{pr_number}: {str(e)}")
        return []

def review_task_func(file_content, file_path):
    """Generate review comments for a file using flake8 and LLM."""
    try:
        issues = run_linter(file_content, file_path)
        if not issues:
            return f"✅ No issues found in {file_path}"

        prompt = (
            "You are a code review assistant for a Python project. Below is a list of issues found by flake8:\n\n"
            f"{chr(10).join(issues)}\n\n"
            f"Code:\n``````\n{file_content}\n``````\n\n"
            "For each issue, provide a clear, concise, and actionable comment explaining the problem and suggesting a fix. "
            "Format your response as a bulleted list with specific line references."
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating review comments for {file_path}: {str(e)}")
        return f"❌ Failed to generate comments for {file_path}: {str(e)}"

def fix_task_func(file_content, issues, file_path):
    """Suggest and apply fixes for identified Python issues."""
    try:
        lines = file_content.splitlines()
        fixed_lines = lines.copy()
        comments = []

        for issue in issues:
            match = re.match(r'.*?:(\d+):(\d+): (\w+) (.*)', issue)
            if match:
                line_num = int(match.group(1)) - 1
                code = match.group(3)
                message = match.group(4)

                if line_num < len(lines):
                    if code == 'F841' and 'undefined name' not in message:  # Unused variable
                        var_match = re.search(r"'([^']+)'", message)
                        if var_match:
                            var_name = var_match.group(1)
                            fixed_lines[line_num] = f"# Removed unused variable: {lines[line_num]}"
                            comments.append(f"Line {line_num + 1}: Removed unused variable '{var_name}'")
                    elif code == 'D100' and 'missing docstring' in message:  # Missing docstring
                        fixed_lines.insert(0, '"""Sample docstring."""')
                        comments.append("Line 1: Added missing docstring")

        return {'fixed_content': '\n'.join(fixed_lines), 'comments': comments}
    except Exception as e:
        logger.error(f"Error suggesting fixes for {file_path}: {str(e)}")
        return {'fixed_content': file_content, 'comments': []}

def create_fix_pr(pr_number, file_path, fixed_content, comments):
    """Create a PR with suggested fixes."""
    if os.getenv("GITHUB_ACTIONS") != "true":
        try:
            import inquirer
            questions = [inquirer.Confirm("create_pr", message=f"Create fix PR for {file_path}?", default=True)]
            answers = inquirer.prompt(questions)
            if not answers or not answers["create_pr"]:
                logger.info(f"Skipped creating fix PR for {file_path}")
                return None
        except ImportError:
            logger.warning("inquirer not available, auto-creating PR")

    try:
        pr = repo.get_pull(pr_number)
        branch_name = f"fix-pr-{pr_number}-{file_path.replace('/', '-').replace('.', '-')}"

        # Create new branch from PR's head
        repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=pr.head.sha
        )

        # Get current file content and SHA
        try:
            current_file = repo.get_contents(file_path, ref=pr.head.ref)
            file_sha = current_file.sha
        except github.GithubException:
            logger.error(f"Could not fetch current file {file_path}")
            return None

        # Update file with fixes
        repo.update_file(
            path=file_path,
            message=f"Automated fixes for PR #{pr_number}: {file_path}",
            content=fixed_content,
            sha=file_sha,
            branch=branch_name
        )

        # Create pull request
        fix_pr = repo.create_pull(
            title=f"🔧 Automated Fixes for PR #{pr_number}: {file_path}",
            body=f"Automated fixes for `{file_path}`:\n\n" + "\n".join([f"- {comment}" for comment in comments]) + f"\n\n🔗 Related to PR #{pr_number}",
            head=branch_name,
            base=pr.base.ref
        )

        logger.info(f"Created fix PR #{fix_pr.number}")
        return fix_pr.number
    except github.GithubException as e:
        logger.error(f"Error creating fix PR for {file_path}: {str(e)}")
        return None

def main():
    """Main function to orchestrate the code review process."""
    try:
        pr_files = get_pr_files(PR_NUMBER)
        if not pr_files:
            logger.info("No files to process. Exiting.")
            return

        # Create CrewAI crew
        crew = Crew(
            agents=[code_reviewer, code_fixer],
            tasks=[],
            verbose=True
        )

        for file in pr_files:
            logger.info(f"Processing {file.filename}")

            try:
                file_content = repo.get_contents(file.filename, ref=repo.get_pull(PR_NUMBER).head.sha).decoded_content.decode()
            except github.GithubException as e:
                logger.error(f"Error fetching content for {file.filename}: {str(e)}")
                continue

            # Execute review task
            review_comments = review_task_func(file_content, file.filename)

            # Post review comments
            try:
                repo.get_pull(PR_NUMBER).create_issue_comment(f"## 🔍 Review for `{file.filename}`\n\n{review_comments}")
                logger.info(f"Posted review comments for {file.filename}")
            except github.GithubException as e:
                logger.error(f"Error posting comment for {file.filename}: {str(e)}")

            # Run linter to get issues for fix task
            issues = run_linter(file_content, file.filename)
            if not issues:
                logger.info(f"No fixable issues found in {file.filename}")
                continue

            # Execute fix task
            fix_result = fix_task_func(file_content, issues, file.filename)
            fixed_content = fix_result.get('fixed_content', file_content)
            fix_comments = fix_result.get('comments', [])

            if fix_comments:
                fix_pr_number = create_fix_pr(PR_NUMBER, file.filename, fixed_content, fix_comments)
                if fix_pr_number:
                    try:
                        repo.get_pull(PR_NUMBER).create_issue_comment(
                            f"## 🔧 Automated Fixes\n\nCreated fix PR #{fix_pr_number} with automated fixes for `{file.filename}`."
                        )
                        logger.info(f"Posted fix PR link for {file.filename}")
                    except github.GithubException as e:
                        logger.error(f"Error posting fix PR comment for {file.filename}: {str(e)}")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()