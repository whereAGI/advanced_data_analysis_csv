"""
Custom Groq LLM implementation for PandasAI.

This module provides a custom LLM class that connects PandasAI with the Groq API.
"""

import re
import requests
import json
import logging
import time  # Added for rate limiting
from collections import deque  # Added for rate limiting
import threading  # Added for thread safety
from typing import Optional, List, Dict, Any, Union
import pandas as pd

# Import PandasAI components
try:
    # Try to import from both PandasAI v1 and v2
    try:
        # PandasAI v2
        from pandasai.llm import LLM
        PANDASAI_V2 = True
    except ImportError:
        # PandasAI v1
        from pandasai.llm.base import LLM
        PANDASAI_V2 = False
    PANDASAI_AVAILABLE = True
except ImportError as e:
    PANDASAI_AVAILABLE = False
    PANDASAI_V2 = False
    # Define a stub class for type hints
    class LLM:
        pass
    # Instead of raising, we'll log the error
    logger.error(f"PandasAI package not found: {e}")
    logger.error("Make sure to install it with: pip install pandasai>=2.0.0")

# Get the logger
logger = logging.getLogger(__name__)

# Rate Limiter Class for Groq API
class GroqTokenRateLimiter:
    def __init__(self, rate_limit_tokens: int = 5500, time_window_seconds: int = 60):
        """
        Initialize a token rate limiter for Groq API.
        
        Args:
            rate_limit_tokens: Maximum tokens allowed per minute (default: 5500, slightly below the 6000 limit)
            time_window_seconds: Time window in seconds (default: 60 seconds)
        """
        self.rate_limit_tokens = rate_limit_tokens
        self.time_window_seconds = time_window_seconds
        self.token_log = deque()  # Stores (timestamp, tokens_used) tuples
        self.lock = threading.Lock()  # Lock for thread safety
        self.debug_log = deque(maxlen=100)  # Store recent debug messages
        self.log_debug_message("Rate limiter initialized")
    
    def log_debug_message(self, message: str):
        """Add a timestamped debug message to the log"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.debug_log.append(f"[{timestamp}] {message}")
        logger.debug(message)
    
    def get_debug_log(self):
        """Return the debug log as a list"""
        return list(self.debug_log)
    
    def _cleanup_log(self):
        """Remove entries older than the time window."""
        current_time = time.monotonic()
        cutoff_time = current_time - self.time_window_seconds
        
        before_count = len(self.token_log)
        # Remove old entries
        while self.token_log and self.token_log[0][0] < cutoff_time:
            oldest_entry = self.token_log.popleft()
            self.log_debug_message(f"Removed expired entry: {oldest_entry[1]} tokens from {time.strftime('%H:%M:%S', time.localtime(oldest_entry[0]))}")
        
        after_count = len(self.token_log)
        if before_count != after_count:
            self.log_debug_message(f"Cleaned up log: removed {before_count - after_count} expired entries")
    
    def get_current_tokens(self) -> int:
        """Calculate total tokens used within current time window."""
        with self.lock:
            self._cleanup_log()
            total = sum(tokens for _, tokens in self.token_log)
            self.log_debug_message(f"Current token usage: {total}/{self.rate_limit_tokens}")
            return total
    
    def get_token_log_summary(self):
        """Get summary of token usage for debugging"""
        with self.lock:
            self._cleanup_log()
            if not self.token_log:
                return "No recent token usage"
            
            entries = [f"{time.strftime('%H:%M:%S', time.localtime(ts))}: {tokens} tokens" 
                      for ts, tokens in self.token_log]
            total = sum(tokens for _, tokens in self.token_log)
            
            return {
                "entries": entries,
                "total": total,
                "limit": self.rate_limit_tokens,
                "entry_count": len(self.token_log)
            }
    
    def log_usage(self, tokens_used: int):
        """Log token usage from a completed API call."""
        if tokens_used is None or tokens_used <= 0:
            self.log_debug_message(f"Skipped logging invalid token count: {tokens_used}")
            return
            
        with self.lock:
            current_time = time.monotonic()
            self.token_log.append((current_time, tokens_used))
            current_total = sum(tokens for _, tokens in self.token_log)
            self.log_debug_message(f"Logged {tokens_used} new tokens. Current window: {current_total}/{self.rate_limit_tokens} tokens")
    
    def wait_if_needed(self):
        """Check rate limit and wait if necessary before proceeding."""
        with self.lock:
            self._cleanup_log()
            current_tokens = sum(tokens for _, tokens in self.token_log)
            
            if current_tokens < self.rate_limit_tokens:
                # Enough capacity to proceed
                available = self.rate_limit_tokens - current_tokens
                self.log_debug_message(f"Rate limit check passed: {current_tokens}/{self.rate_limit_tokens} tokens used in last minute (Available: {available} tokens)")
                return
                
            # Need to wait - calculate delay based on oldest entry
            if self.token_log:
                oldest_timestamp, oldest_tokens = self.token_log[0]
                # Calculate when the oldest entry will expire
                now = time.monotonic()
                expiry_time = oldest_timestamp + self.time_window_seconds
                wait_time = max(0.1, expiry_time - now + 0.1)  # Add small buffer and ensure positive
                expiry_time_str = time.strftime('%H:%M:%S', time.localtime(expiry_time))
                
                self.log_debug_message(f"⚠️ Rate limit reached: {current_tokens}/{self.rate_limit_tokens} tokens used. " +
                              f"Waiting {wait_time:.2f}s for {oldest_tokens} tokens to expire at {expiry_time_str}")
                
                # Log all entries for better debugging
                for i, (ts, tokens) in enumerate(self.token_log):
                    ts_str = time.strftime('%H:%M:%S', time.localtime(ts))
                    expiry = time.strftime('%H:%M:%S', time.localtime(ts + self.time_window_seconds))
                    self.log_debug_message(f"   Token entry {i+1}: {tokens} tokens used at {ts_str}, expires at {expiry}")
                
                # Wait for the earliest tokens to expire with progress updates
                wait_start = time.monotonic()
                wait_end = wait_start + wait_time
                
                # Print initial wait message
                logger.info(f"Rate limit waiting: 0/{wait_time:.1f}s - Next tokens expire at {expiry_time_str}")
                
                # Wait with progress updates every 5 seconds
                while time.monotonic() < wait_end:
                    elapsed = time.monotonic() - wait_start
                    remaining = wait_time - elapsed
                    
                    # Sleep in small increments (5 seconds max)
                    sleep_time = min(5.0, remaining)
                    if sleep_time <= 0:
                        break
                        
                    time.sleep(sleep_time)
                    
                    # Update progress after each sleep increment
                    elapsed = time.monotonic() - wait_start
                    logger.info(f"Rate limit waiting: {elapsed:.1f}/{wait_time:.1f}s - {(elapsed/wait_time*100):.0f}% complete")
                
                # Final check before proceeding
                self._cleanup_log()
                current_tokens = sum(tokens for _, tokens in self.token_log)
                self.log_debug_message(f"After waiting, token usage is now: {current_tokens}/{self.rate_limit_tokens}")
                
                # If still over limit, return anyway to avoid infinite recursion
                # The caller should handle retries appropriately
                if current_tokens >= self.rate_limit_tokens:
                    self.log_debug_message("⚠️ Still at rate limit after waiting, but proceeding to avoid deadlock")
            else:
                # This shouldn't happen but just in case
                self.log_debug_message("Warning: Rate limit reached but no token log entries found")

class GroqLLM(LLM):
    """
    Custom LLM implementation for Groq
    """
    def __init__(
        self, 
        api_token: Optional[str] = None,
        model: str = "deepseek-r1-distill-llama-70b",  # Using Deepseek model as default
        temperature: float = 0.1,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize the Groq LLM
        
        Args:
            api_token: Groq API token
            model: Model name to use (default: deepseek-r1-distill-llama-70b)
            temperature: Temperature for generation (default: 0.1)
            max_tokens: Maximum tokens to generate (default: 1000)
        """
        # Save parameters
        self.api_token = api_token
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = "https://api.groq.com/openai/v1/chat/completions"
        self.last_prompt = ""
        
        # Add tracking variables for debugging UI
        self.last_query = None
        self.last_token_usage = None
        self.last_query_time = None
        self.last_thinking = None
        
        # Add rate limiter
        self.rate_limiter = GroqTokenRateLimiter()
        self.limiter_lock = threading.Lock()
        
        # Call parent constructor
        super().__init__()
        
        # Log initialization
        logger.info(f"Initialized GroqLLM with model: {self.model}")
    
    @property
    def type(self) -> str:
        """
        Get the type of the LLM
        
        Returns:
            The type identifier for this LLM
        """
        return "groq"
    
    def get_models(self) -> List[str]:
        """
        Get available models
        
        Returns:
            List of available model names
        """
        return [
            "deepseek-r1-distill-llama-70b",
            "qwen-qwq-32b",
            "meta-llama/llama-4-scout-17b-16e-instruct"
        ]
    
    def get_debug_info(self):
        """Get debugging information about the rate limiter status and query details"""
        debug_info = {
            "log": [],
            "token_usage": None,
            "model": self.model,
            "api_base": self.api_base,
            "last_query": self.last_query,
            "last_token_usage": self.last_token_usage,
            "last_query_time": self.last_query_time
        }
        
        if hasattr(self, 'rate_limiter'):
            debug_info["log"] = self.rate_limiter.get_debug_log()
            debug_info["token_usage"] = self.rate_limiter.get_token_log_summary()
            
        return debug_info
    
    def call(self, prompt: str, max_tokens: int = None) -> str:
        """
        Call the Groq API
        
        Args:
            prompt: The prompt to send to the API
            max_tokens: Optional override for max tokens (default: use instance value)
            
        Returns:
            Response from the API
        """
        if not self.api_token:
            raise ValueError("API token not provided")
            
        # Store the prompt for later use
        self.last_prompt = prompt
        self.last_query = None  # Reset query tracking
        
        # Use provided max_tokens or fall back to instance default
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Log the API request
        logger.info(f"Calling Groq API with model: {self.model}")
        logger.debug(f"Prompt: {prompt}")
            
        # Create request payload
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": tokens
        }
            
        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }
            
        # Global retry counter to avoid infinite loops
        max_retries = 5
        retry_count = 0
        backoff_time = 5  # Start with 5 seconds backoff
        
        # Call API with rate limiting and retries
        while retry_count < max_retries:
            try:
                # Apply rate limiting before making the request
                with self.limiter_lock:
                    logger.info("Checking rate limit before API call")
                    # Get current token usage before the call
                    current_tokens = self.rate_limiter.get_current_tokens()
                    available_tokens = self.rate_limiter.rate_limit_tokens - current_tokens
                    logger.info(f"Token usage before call: {current_tokens}/{self.rate_limiter.rate_limit_tokens} (Available: {available_tokens})")
                    
                    # Wait if needed based on rate limit
                    self.rate_limiter.wait_if_needed()
                
                    # Make the API call with improved error handling
                    request_time = time.time()
                    self.rate_limiter.log_debug_message(f"Making API call to {self.api_base} with model {self.model}")
                    
                    try:
                        response = requests.post(
                            self.api_base, 
                            headers=headers, 
                            data=json.dumps(payload),
                            timeout=60  # Add timeout to prevent hanging
                        )
                        
                        # Check for rate limit errors specifically
                        if response.status_code == 429:
                            retry_after = response.headers.get('Retry-After', '60')
                            try:
                                retry_after = int(retry_after)
                            except ValueError:
                                retry_after = 60  # Default to 60 seconds if not a valid number
                                
                            error_msg = f"Rate limit exceeded (429). Retry after: {retry_after} seconds"
                            self.rate_limiter.log_debug_message(f"Rate limit error: {error_msg}")
                            logger.error(error_msg)
                            
                            # Increment retry counter
                            retry_count += 1
                            
                            # Wait longer for rate limit to reset
                            wait_time = max(retry_after, backoff_time)
                            logger.info(f"Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                            
                            # Force a reset of the rate limiter log for the next minute
                            self.rate_limiter.token_log.clear()
                            self.rate_limiter.log_debug_message("Cleared token log due to rate limit error")
                            
                            # Add a penalty entry to prevent immediate retries
                            # Instead of full penalty, use 80% of limit with a short expiry time
                            penalty_tokens = int(self.rate_limiter.rate_limit_tokens * 0.8)
                            
                            # Set a shorter expiry for the penalty by backdating the timestamp
                            backdate_seconds = self.rate_limiter.time_window_seconds - retry_after
                            if backdate_seconds < 0:
                                backdate_seconds = 0
                                
                            # Add a backdated penalty entry so it expires after retry_after seconds
                            with self.rate_limiter.lock:
                                current_time = time.monotonic() - backdate_seconds
                                self.rate_limiter.token_log.append((current_time, penalty_tokens))
                                
                            self.rate_limiter.log_debug_message(f"Added penalty of {penalty_tokens} tokens with {retry_after}s expiry to enforce backoff")
                            
                            # Wait and then continue to the next iteration (retry)
                            wait_start = time.monotonic()
                            wait_end = wait_start + wait_time
                            
                            # Print initial wait message
                            logger.info(f"Rate limit (HTTP 429) - Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                            logger.info(f"Wait progress: 0/{wait_time:.1f}s - 0% complete")
                            
                            # Wait with progress updates every 5 seconds
                            while time.monotonic() < wait_end:
                                elapsed = time.monotonic() - wait_start
                                remaining = wait_time - elapsed
                                
                                # Sleep in small increments (5 seconds max)
                                sleep_time = min(5.0, remaining)
                                if sleep_time <= 0:
                                    break
                                    
                                time.sleep(sleep_time)
                                
                                # Update progress after each sleep increment
                                elapsed = time.monotonic() - wait_start
                                progress_pct = min(100, int(elapsed/wait_time*100))
                                logger.info(f"Wait progress: {elapsed:.1f}/{wait_time:.1f}s - {progress_pct}% complete")
                            
                            # Final progress update
                            logger.info(f"Wait complete - Retrying API call (attempt {retry_count}/{max_retries})")
                            
                            backoff_time *= 2  # Exponential backoff
                            continue
                            
                        # Check for other errors
                        response.raise_for_status()
                        
                        # Parse response
                        response_json = response.json()
                        
                        # Calculate response time
                        response_time = time.time() - request_time
                        self.rate_limiter.log_debug_message(f"API call completed in {response_time:.2f}s")
                        
                        # Save query details for debugging UI
                        self.last_query_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        self.last_query = prompt  # Store the query for UI debugging
                        
                        if "usage" in response_json:
                            usage = response_json["usage"]
                            if "completion_tokens" in usage:
                                completion_tokens = usage["completion_tokens"]
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                total_tokens = usage.get("total_tokens", 0)
                                
                                # Store token usage for UI
                                self.last_token_usage = {
                                    "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": total_tokens,
                                    "timestamp": self.last_query_time
                                }
                                
                                self.rate_limiter.log_debug_message(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                                # Log the completion tokens for rate limiting
                                self.rate_limiter.log_usage(completion_tokens)
                                
                                # Update current usage after this call
                                current_usage = self.rate_limiter.get_current_tokens()
                                self.rate_limiter.log_debug_message(f"Updated token usage: {current_usage}/{self.rate_limiter.rate_limit_tokens} tokens in the last minute")
                            else:
                                self.rate_limiter.log_debug_message("No completion_tokens found in usage data")
                        else:
                            self.rate_limiter.log_debug_message("No usage information found in response")
                        
                        # Extract the result
                        result = response_json["choices"][0]["message"]["content"]
                        
                        # Success - return the result
                        return result
                    
                    except requests.exceptions.RequestException as e:
                        # Handle all request exceptions including timeouts, connection errors, etc.
                        error_message = str(e)
                        self.rate_limiter.log_debug_message(f"API request error: {error_message}")
                        
                        # Add specific handling for rate limit errors that might be caught as general exceptions
                        if "429" in error_message or "Too Many Requests" in error_message:
                            # Increment retry counter
                            retry_count += 1
                            
                            # Force a reset of the rate limiter log
                            self.rate_limiter.token_log.clear()
                            self.rate_limiter.log_debug_message("Cleared token log due to general exception rate limit error")
                            
                            # Add a partial penalty with shorter expiry time
                            penalty_tokens = int(self.rate_limiter.rate_limit_tokens * 0.8)
                            # Set timestamp to expire after wait_time
                            wait_time = backoff_time
                            backdate_seconds = max(0, self.rate_limiter.time_window_seconds - wait_time)
                            current_time = time.monotonic() - backdate_seconds
                            
                            # Add the backdated entry directly
                            with self.rate_limiter.lock:
                                self.rate_limiter.token_log.append((current_time, penalty_tokens))
                            
                            self.rate_limiter.log_debug_message(f"Added penalty of {penalty_tokens} tokens with {wait_time}s expiry")
                            
                            # Wait with progress updates
                            wait_start = time.monotonic()
                            wait_end = wait_start + wait_time
                            
                            # Print initial wait message
                            logger.info(f"Rate limit (general exception) - Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                            logger.info(f"Wait progress: 0/{wait_time:.1f}s - 0% complete")
                            
                            # Wait with progress updates every 5 seconds
                            while time.monotonic() < wait_end:
                                elapsed = time.monotonic() - wait_start
                                remaining = wait_time - elapsed
                                
                                # Sleep in small increments (5 seconds max)
                                sleep_time = min(5.0, remaining)
                                if sleep_time <= 0:
                                    break
                                    
                                time.sleep(sleep_time)
                                
                                # Update progress after each sleep increment
                                elapsed = time.monotonic() - wait_start
                                progress_pct = min(100, int(elapsed/wait_time*100))
                                logger.info(f"Wait progress: {elapsed:.1f}/{wait_time:.1f}s - {progress_pct}% complete")
                            
                            # Final progress update
                            logger.info(f"Wait complete - Retrying API call (attempt {retry_count}/{max_retries})")
                            
                            backoff_time *= 2  # Exponential backoff
                            continue
                        
                        # Re-raise non-rate-limit exceptions
                        raise
            
            except Exception as e:
                # Increment retry counter for all other exceptions too
                retry_count += 1
                
                error_message = str(e)
                # Log the error
                logger.error(f"Error calling Groq API: {error_message}")
                
                # Track failure for debugging UI
                self.last_query_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.last_query = prompt
                self.last_token_usage = {"error": error_message}
                
                # For rate limit errors, retry with backoff
                if "429" in error_message or "Too Many Requests" in error_message:
                    logger.info(f"Rate limit hit, retry {retry_count}/{max_retries}")
                    
                    # Force a reset of the rate limiter log
                    self.rate_limiter.token_log.clear()
                    self.rate_limiter.log_debug_message("Cleared token log due to general exception rate limit error")
                    
                    # Add a partial penalty with shorter expiry time
                    penalty_tokens = int(self.rate_limiter.rate_limit_tokens * 0.8)
                    # Set timestamp to expire after wait_time
                    wait_time = backoff_time
                    backdate_seconds = max(0, self.rate_limiter.time_window_seconds - wait_time)
                    current_time = time.monotonic() - backdate_seconds
                    
                    # Add the backdated entry directly
                    with self.rate_limiter.lock:
                        self.rate_limiter.token_log.append((current_time, penalty_tokens))
                    
                    self.rate_limiter.log_debug_message(f"Added penalty of {penalty_tokens} tokens with {wait_time}s expiry")
                    
                    # Wait with progress updates
                    wait_start = time.monotonic()
                    wait_end = wait_start + wait_time
                    
                    # Print initial wait message
                    logger.info(f"Rate limit (general exception) - Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                    logger.info(f"Wait progress: 0/{wait_time:.1f}s - 0% complete")
                    
                    # Wait with progress updates every 5 seconds
                    while time.monotonic() < wait_end:
                        elapsed = time.monotonic() - wait_start
                        remaining = wait_time - elapsed
                        
                        # Sleep in small increments (5 seconds max)
                        sleep_time = min(5.0, remaining)
                        if sleep_time <= 0:
                            break
                            
                        time.sleep(sleep_time)
                        
                        # Update progress after each sleep increment
                        elapsed = time.monotonic() - wait_start
                        progress_pct = min(100, int(elapsed/wait_time*100))
                        logger.info(f"Wait progress: {elapsed:.1f}/{wait_time:.1f}s - {progress_pct}% complete")
                    
                    # Final progress update
                    logger.info(f"Wait complete - Retrying API call (attempt {retry_count}/{max_retries})")
                    
                    backoff_time *= 2  # Exponential backoff
                    continue
                
                # Re-raise non-rate-limit exceptions
                raise
            
            # If we got here, the API call was successful
            break
        
        # If we exhausted all retries
        if retry_count >= max_retries:
            error_msg = f"Maximum retries ({max_retries}) exceeded for API call due to rate limits"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # This should not be reached as we either return in the try block or raise in the except block
        return result
    
    def _preprocess_response(self, text: str) -> str:
        """
        Preprocess the raw response to extract clean code
        
        Args:
            text: Raw model response
            
        Returns:
            Cleaned code-only response
        """
        try:
            # Safety check for invalid input
            if not isinstance(text, str):
                logger.error(f"Invalid response type: {type(text)}")
                return ""
                
            # Fix malformed markdown fences (4 backticks instead of 3)
            text = re.sub(r'^`{3,4}', '```', text)
            text = re.sub(r'`{3,4}$', '```', text)
            
            # Replace markdown fence with no language specifier
            text = re.sub(r'```(\s*\n)', '```python\n', text)
                
            # Check for thinking section format: <think>...</think> followed by Python code
            thinking_pattern = r"<think>(.*?)</think>\s*(?:Python code:|python code:)(.*)"
            thinking_match = re.search(thinking_pattern, text, re.DOTALL)
            if thinking_match:
                logger.info("Found thinking section in response, extracting code part...")
                # Store thinking for debugging
                self.last_thinking = thinking_match.group(1).strip()
                return thinking_match.group(2).strip()
            
            # Check for dictionary format
            dict_pattern = r"{'type':\s*'code',\s*'value':\s*'(.*?)'}"
            dict_match = re.search(dict_pattern, text, re.DOTALL)
            if dict_match:
                logger.info("Found dictionary format in response, extracting code...")
                return dict_match.group(1).replace('\\n', '\n')
            
            # Look for Python code marker (case insensitive)
            if re.search(r"(?:Python|python) code:", text):
                logger.info("Found 'Python code:' marker, extracting code...")
                code_parts = re.split(r"(?:Python|python) code:", text, 1)
                if len(code_parts) > 1:
                    return code_parts[1].strip()
            
            # Extract code from markdown code blocks if present - including handling end of string
            code_pattern = r"```(?:python)?\s*([\s\S]*?)(?:```|\Z)"
            matches = re.findall(code_pattern, text)
            if matches:
                logger.info("Found code blocks, extracting first code block...")
                return matches[0].strip()
            
            # If we detect any thinking pattern in the response, extract just proper code
            if "<think>" in text or "thinking" in text.lower() or "let me" in text.lower():
                logger.info("Detected thinking text, extracting clean code...")
                # Try to find the end of thinking section and the start of actual code
                lines = text.split('\n')
                code_lines = []
                in_code_section = False
                
                for i, line in enumerate(lines):
                    # Skip lines that are clearly part of thinking or markdown
                    if '```' in line or line.startswith('#') or line.startswith('>'):
                        continue
                        
                    # Look for indicators of actual code
                    if not in_code_section:
                        if ("import" in line and "pandas" in line) or "result =" in line:
                            in_code_section = True
                            code_lines.append(line)
                        elif i > 0 and i < len(lines) - 1:
                            # Look for indented code blocks
                            prev_line = lines[i-1].strip()
                            if (prev_line.endswith(':') and (line.startswith('    ') or line.startswith('\t'))):
                                in_code_section = True
                                code_lines.append(lines[i-1])  # Include the line with the colon
                                code_lines.append(line)
                    else:
                        # We're in a code section
                        # Check if we've reached the end of the code block
                        if not line.strip() and i < len(lines) - 1:
                            # If next non-empty line doesn't look like code, end the block
                            next_code_line = None
                            for j in range(i+1, min(i+5, len(lines))):
                                if lines[j].strip():
                                    next_code_line = lines[j]
                                    break
                                    
                            if next_code_line and not (
                                next_code_line.startswith('    ') or 
                                next_code_line.startswith('\t') or
                                "=" in next_code_line or
                                "import" in next_code_line or
                                "def " in next_code_line
                            ):
                                break
                        code_lines.append(line)
                
                if code_lines:
                    return '\n'.join(code_lines)
            
            # Fallback to returning original text after basic cleanup
            # Remove markdown artifacts
            text = re.sub(r'```python|```', '', text)
            
            # Check and attempt to fix unterminated strings
            lines = []
            open_quote = None
            
            for line in text.split('\n'):
                for char in line:
                    if char in ['"', "'"]:
                        if open_quote is None:
                            open_quote = char
                        elif char == open_quote:
                            open_quote = None
                
                # If line ends with unclosed quote, close it
                if open_quote:
                    line += open_quote
                    open_quote = None
                
                lines.append(line)
            
            return '\n'.join(lines)
            
        except Exception as e:
            logger.error(f"Error preprocessing response: {e}")
            return text
    
    def generate_code(self, prompt: str, pipeline_context=None) -> str:
        """
        Generate code from a prompt
        
        Args:
            prompt: The prompt to generate code from
            pipeline_context: PandasAI pipeline context (optional)
            
        Returns:
            Generated code as string
        """
        try:
            # Create a coding-specific prompt
            coding_prompt = f"""
            You are an AI coding assistant. Generate Python code to answer this question: {prompt}
            
            Important:
            1. Your code should create a 'result' variable with format: {{"type": "dataframe", "value": df}}
            2. Generate ONLY executable Python code
            3. Do NOT include thinking steps in your final output
            4. It's okay to think about the solution, but put your thinking in <think> tags, which I'll remove
            5. After your thinking, start with "Python code:" and then write the final code
            6. The final code must be pure Python without any markdown
            7. Handle all edge cases
            8. Return empty DataFrame if no results found
            9. Include RELATED data that provides context to the primary data
            10. Look for relationships in column names and data patterns
            11. When retrieving specific data items, include relevant attributes that enhance understanding
            12. Consider creating joins if multiple tables are referenced in the schema
            
            First think through your approach, then provide executable code:
            """
            
            # Store this as the last query for UI debugging
            self.last_query = coding_prompt
            
            # Call API
            result = self.call(coding_prompt)
            
            # Try to extract thinking sections
            thinking_match = re.search(r"<think>(.*?)</think>", result, re.DOTALL)
            if thinking_match:
                self.last_thinking = thinking_match.group(1).strip()
            
            # At this point, result should be clean code, but let's ensure it
            code = self._extract_code(result)
            
            if not code or not isinstance(code, str):
                logger.warning("No valid code found in API response")
                # Ensure we return a valid string
                code = "import pandas as pd\nresult = {\"type\": \"dataframe\", \"value\": pd.DataFrame()}"
            
            # Clean up the code string
            code = code.strip()
            
            # Log the generated code for debugging
            logger.info(f"Code generated:\n{code}")
            
            # Return code as string
            return code
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            # Return a safe fallback
            return "import pandas as pd\nresult = {\"type\": \"dataframe\", \"value\": pd.DataFrame()}"
    
    def generate_pandas_code(self, prompt: str, df_name: str = "df", pipeline_context=None) -> str:
        """
        Generate pandas-specific code from a prompt
        
        Args:
            prompt: The prompt to generate code from
            df_name: The name of the DataFrame variable
            pipeline_context: PandasAI pipeline context (optional)
            
        Returns:
            Generated pandas code as string
        """
        try:
            # Create a pandas-specific prompt
            pandas_prompt = f"""
            Given a pandas DataFrame called '{df_name}', generate Python code to answer this question: {prompt}
            
            Important:
            1. Your code must use only the DataFrame '{df_name}' and return a result variable
            2. Generate ONLY executable Python code
            3. Handle all edge cases and missing data
            4. Include related data that provides context to the primary data
            5. Include columns that help understand the query results better
            6. Look for relationships between columns when returning data
            7. Consider including additional context columns beyond just the exact query
            
            The format of the result must be:
            result = {{"type": "dataframe", "value": answer_df}}
            
            Example:
            ```python
            # Filter data
            filtered_data = {df_name}[{df_name}['column'] > value]
            
            # Add related contextual information
            if 'related_column' in {df_name}.columns:
                result_df = filtered_data[['column', 'related_column', 'context_column']]
            else:
                result_df = filtered_data[['column']]
                
            result = {{"type": "dataframe", "value": result_df}}
            ```
            
            Please think step by step. Put your thinking steps in <think></think> tags and only include the final code outside the tags.
            """
            
            # Store this as the last query for UI debugging
            self.last_query = pandas_prompt
            
            # Call the API
            result = self.call(pandas_prompt)
            
            # Extract code from response
            code = self._extract_code(result)
            
            if not code or not isinstance(code, str):
                logger.warning("No valid pandas code found in API response")
                # Ensure we return a valid string
                code = f"import pandas as pd\nresult = {{\"type\": \"dataframe\", \"value\": pd.DataFrame()}}"
            
            # Clean up the code string
            code = code.strip()
            
            # Log the generated code for debugging
            logger.info(f"Generated pandas code:\n{code}")
            
            # Return code as string
            return code
            
        except Exception as e:
            logger.error(f"Error generating pandas code: {e}")
            # Return a safe fallback
            return f"import pandas as pd\nresult = {{\"type\": \"dataframe\", \"value\": pd.DataFrame()}}"
    
    def analyze_dataframe(self, df: pd.DataFrame, query: str, schema_context: str) -> Union[str, Dict[str, str]]:
        """
        Analyze a DataFrame in context of a query and schema
        
        Args:
            df: DataFrame to analyze
            query: The query that generated this DataFrame
            schema_context: Schema context to help with analysis
            
        Returns:
            Analysis as string or dict with analysis and reasoning
        """
        try:
            # Create a description of the DataFrame
            df_desc = f"DataFrame with {len(df)} rows and {len(df.columns)} columns: {', '.join(df.columns)}"
            
            # Calculate statistics using the FULL dataset
            # Calculate some basic statistics if numeric columns exist
            stats = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats[col] = {
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "median": float(df[col].median()),
                        "sum": float(df[col].sum()),
                        "std": float(df[col].std())
                    }
            
            # Get full frequency counts for categorical columns
            counts = {}
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 50:
                    # Convert to regular Python types for JSON serialization
                    value_counts = df[col].value_counts().to_dict()
                    counts[col] = {str(k): int(v) for k, v in value_counts.items()}
            
            # Prepare a sample of data with proper serialization for timestamps
            def json_serialize_df(dataframe):
                # Convert DataFrame to dict with proper timestamp handling
                df_dict = dataframe.copy()
                
                # Convert all timestamps to ISO format strings for JSON compatibility
                for c in df_dict.columns:
                    if pd.api.types.is_datetime64_dtype(df_dict[c]):
                        df_dict[c] = df_dict[c].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                return json.dumps(df_dict.to_dict(orient='records'), default=str)
            
            # More accurate token estimation
            # For a better estimation of tokens based on actual data content
            TOKEN_LIMIT = 4000  # Configurable token limit, adjust as needed
            
            # Function to better estimate token count
            def estimate_tokens(dataframe):
                # Serialize dataframe to get actual string length
                serialized = json_serialize_df(dataframe)
                # Rough estimate - 1 token is ~4 characters in English text
                char_count = len(serialized)
                return char_count // 4
            
            # Estimate tokens for the full dataframe
            estimated_tokens = estimate_tokens(df)
            data_too_large = estimated_tokens > TOKEN_LIMIT
            
            logger.info(f"DataFrame size: {len(df)} rows × {len(df.columns)} columns, estimated tokens: {estimated_tokens}")
            
            # Determine how much data to include based on size
            if data_too_large:
                # If data is too large, try to find the largest sample that fits
                logger.info(f"Data too large for full analysis (est. {estimated_tokens} tokens > {TOKEN_LIMIT} limit)")
                
                # Try to include as many rows as possible within the token limit
                sample_size = 5  # Start with minimum 5 rows
                for test_size in [10, 25, 50, 100]:  # Test progressively larger samples
                    if estimate_tokens(df.head(test_size)) < TOKEN_LIMIT * 0.7:  # Use 70% of token limit
                        sample_size = test_size
                
                # Include detailed message about data size limitation
                data_str = f"""The complete dataset is too large to analyze directly ({len(df)} rows, est. {estimated_tokens} tokens).
Analysis is based on statistics from the full dataset and a sample of {sample_size} rows."""
                sample_str = f"Sample ({sample_size} rows for reference):\n{json_serialize_df(df.head(sample_size))}"
                
                # Add warning to logs
                logger.warning(f"DataFrame too large for full analysis. Using {sample_size} row sample.")
            else:
                # Include the full dataset in string format
                data_str = f"Full dataset included for comprehensive analysis ({len(df)} rows)."
                sample_str = json_serialize_df(df)
                logger.info(f"Using complete dataset for analysis ({len(df)} rows).")
            
            # Create a prompt for analysis
            analysis_prompt = f"""
            Generate a detailed, data-driven analysis of this DataFrame that was retrieved in response to the query: "{query}"
            
            DataFrame Description: {df_desc}
            
            Data Status: {data_str}
            {sample_str}
            
            Comprehensive Statistics (calculated from the ENTIRE dataset):
            Numeric Column Statistics:
            {json.dumps(stats, indent=2, default=str)}
            
            Value Counts for Categorical Columns (from ENTIRE dataset):
            {json.dumps(counts, indent=2, default=str)}
            
            Schema Context:
            {schema_context}
            
            Your analysis must include:
            1. HIGHLY SPECIFIC insights based on the statistics from the full dataset
            2. Direct references to exact numbers, percentages, and key values from the statistics
            3. Analysis of top values based on the provided frequency counts
            4. Identification of any outliers or notable patterns visible in the statistics
            5. Context-specific insights related directly to the original query: "{query}"
            6. Important! If the user asks for a list or data then actually share the data in a table format in your analysis.
            7. Always query the date and time column if available, this will help you set data in the correct context.
            
            IMPORTANT: Focus on the actual data in the result rather than making assumptions about how it was filtered.
            The data shown is the RESULT of the query - analyze what's present in the data itself.
            
            DO NOT provide generic observations that could apply to any dataset.
            ONLY mention insights that are directly supported by the provided statistics.
            
            First, put your reasoning in <think></think> tags where you analyze the data step by step.
            Then provide a concise, well-structured analysis with concrete, data-specific insights.
            """
            
            # Call the API with increased token limit for analysis and handle rate limit retries
            logger.info("Generating DataFrame analysis...")
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    analysis = self.call(analysis_prompt, max_tokens=3000)
                    break  # Success - exit the retry loop
                except Exception as e:
                    error_message = str(e)
                    # If this is a rate limit error, retry after waiting
                    if "429" in error_message or "Too Many Requests" in error_message or "rate limit" in error_message.lower():
                        retry_count += 1
                        wait_time = 60 * retry_count  # Progressively longer waits
                        logger.warning(f"Rate limit hit during DataFrame analysis. Waiting {wait_time}s before retry {retry_count}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Other error - raise it
                        raise
            else:
                # If we exhausted all retries
                return {
                    "analysis": "Unable to analyze the data due to persistent rate limits. Please try again later.",
                    "reasoning": "Rate limit errors prevented analysis completion after multiple retries."
                }
            
            # Extract reasoning and analysis parts
            result = {"analysis": "", "reasoning": ""}
            
            # If data was too large, note this in the analysis and result
            if data_too_large:
                result["data_truncated"] = True
                result["full_size"] = len(df)
                result["sample_size"] = sample_size
                result["estimated_tokens"] = estimated_tokens
                result["token_limit"] = TOKEN_LIMIT
                data_size_note = f"\n\n**Note:** This analysis is based on statistics from the complete dataset ({len(df)} rows) and a sample of {sample_size} rows, as the full data was too large to analyze directly."
            else:
                result["data_truncated"] = False
                data_size_note = ""
            
            # Check for thinking/reasoning section
            thinking_pattern = r"<think>(.*?)</think>(.*)"
            thinking_match = re.search(thinking_pattern, analysis, re.DOTALL)
            
            if thinking_match:
                # Extract the parts
                result["reasoning"] = thinking_match.group(1).strip()
                result["analysis"] = thinking_match.group(2).strip() + data_size_note
                logger.info("Extracted reasoning and analysis sections")
            else:
                # No thinking section found, use the whole response as analysis
                result["analysis"] = analysis + data_size_note
                logger.info("No reasoning section found, using complete response as analysis")
            
            return result
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Error analyzing DataFrame: {error_message}")
            
            # If this is a rate limit error, provide a clearer error message
            if "429" in error_message or "Too Many Requests" in error_message or "rate limit" in error_message.lower():
                return {
                    "analysis": "Unable to analyze the data due to Groq API rate limits. The API allows a limited number of tokens per minute. Please wait a moment before trying again.",
                    "reasoning": f"Rate limit error: {error_message}"
                }
            
            return {"analysis": f"Unable to analyze the data due to an error: {str(e)}", "reasoning": ""}
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code from LLM response
        
        Args:
            text: Text containing code
            
        Returns:
            Extracted code
        """
        try:
            if not isinstance(text, str):
                logger.warning(f"Input to _extract_code is not a string: {type(text)}")
                return ""
                
            # Clean any special characters that might cause issues
            # Replace any triple backticks at the beginning that could be malformed
            text = re.sub(r'^`{3,4}', '```', text)
            
            # Handle the case where the markdown fence appears with no language specifier
            text = re.sub(r'```(\s*\n)', '```python\n', text)
            
            # Check for the specific format returned by reasoning models
            thinking_pattern = r"<think>(.*?)</think>\s*(?:Python code:|python code:)(.*)"
            thinking_match = re.search(thinking_pattern, text, re.DOTALL)
            if thinking_match:
                logger.info("Found thinking section in response, extracting code part...")
                code = thinking_match.group(2).strip()
                return self._clean_code(code)
            
            # Check for the embedded dictionary format
            dict_pattern = r"{'type':\s*'code',\s*'value':\s*'(.*?)'}"
            dict_match = re.search(dict_pattern, text, re.DOTALL)
            if dict_match:
                logger.info("Found dictionary format in response, extracting code...")
                code = dict_match.group(1).replace('\\n', '\n')
                return self._clean_code(code)
                
            # Try to match code blocks with markdown formatting - handle both ``` and ```` variants
            code_pattern = r"```(?:python)?\s*([\s\S]*?)(?:```|\Z)"
            matches = re.findall(code_pattern, text)
            
            if matches:
                code = matches[0].strip()
                logger.info("Extracted code from markdown code block")
                return self._clean_code(code)
                
            # Look for "Python code:" marker 
            if re.search(r"(?:Python|python) code:", text):
                code_parts = re.split(r"(?:Python|python) code:", text, 1)
                if len(code_parts) > 1:
                    code = code_parts[1].strip()
                    logger.info("Extracted code after 'Python code:' marker")
                    return self._clean_code(code)
            
            # As a fallback, try to find code by looking for common Python imports/patterns
            lines = text.split('\n')
            code_lines = []
            in_code_section = False
            
            for line in lines:
                # Skip lines that are clearly markdown or explanatory text
                if '```' in line or '####' in line or line.startswith('> '):
                    continue
                    
                # Try to detect the start of Python code
                if not in_code_section:
                    if (re.search(r"^import\s+\w+", line) or 
                        re.search(r"^from\s+\w+\s+import", line) or
                        "def " in line or 
                        "result =" in line or
                        "=" in line and not "==" in line and not "!=" in line):
                        in_code_section = True
                        code_lines.append(line)
                else:
                    # Stop collecting when we hit explanatory text again
                    if line.strip() == "" and len(code_lines) > 0:
                        # If we have a blank line after collecting some code, peek ahead
                        # to see if the next non-blank line looks like code
                        continue_collecting = False
                        for peek_line in lines[lines.index(line)+1:]:
                            if peek_line.strip() == "":
                                continue
                            if (re.search(r"^import\s+\w+", peek_line) or 
                                re.search(r"^from\s+\w+\s+import", peek_line) or
                                "def " in peek_line or 
                                "=" in peek_line and not "==" in peek_line and not "!=" in peek_line or
                                peek_line.startswith("    ") or 
                                peek_line.startswith("\t")):
                                continue_collecting = True
                            break
                        
                        if not continue_collecting:
                            break
                code_lines.append(line)
            
            if code_lines:
                logger.info("Extracted code by heuristic pattern matching")
                return self._clean_code('\n'.join(code_lines))
                
            # If we couldn't extract meaningful code, return an empty string
            logger.warning("Could not extract valid code from response")
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting code: {e}")
            return ""
    
    def _clean_code(self, code: str) -> str:
        """
        Clean extracted code to remove artifacts and ensure it's valid Python
        
        Args:
            code: Raw extracted code
            
        Returns:
            Cleaned code
        """
        if not code:
            return ""
            
        # Remove any remaining markdown artifacts
        code = code.replace("```python", "").replace("```", "")
        
        # Fix common issues with quotes and string literals
        lines = []
        open_quote = None
        
        for line in code.split('\n'):
            # Check for unmatched quotes that might cause syntax errors
            for char in line:
                if char in ["'", '"'] and (open_quote is None or char == open_quote):
                    if open_quote is None:
                        open_quote = char
                    else:
                        open_quote = None
            
            # If we have an unmatched quote at the end of the line, close it
            if open_quote is not None:
                line += open_quote
                open_quote = None
                
            lines.append(line)
            
        cleaned_code = '\n'.join(lines)
        
        # Try to validate the code by parsing it - if it fails, fall back to a safer version
        try:
            import ast
            ast.parse(cleaned_code)
            return cleaned_code
        except SyntaxError as e:
            logger.warning(f"Cleaned code still has syntax error: {e}")
            # Fall back to a safer approach by trying to extract just valid Python statements
            return self._extract_safe_code(cleaned_code)
    
    def _extract_safe_code(self, code: str) -> str:
        """
        Extract only valid Python statements from potentially invalid code
        
        Args:
            code: Potentially invalid code
            
        Returns:
            Safe, executable code
        """
        import ast
        
        lines = code.split('\n')
        valid_lines = []
        current_block = []
        
        for line in lines:
            current_block.append(line)
            block_text = '\n'.join(current_block)
            
            try:
                # Try to parse the current block
                ast.parse(block_text)
                # If we get here, the block is valid
                valid_lines.extend(current_block)
                current_block = []
            except SyntaxError:
                # If this is a single line and it has a syntax error, skip it
                if len(current_block) == 1:
                    current_block = []
                # Otherwise keep collecting lines to see if we can complete the statement
        
        # If we have any remaining lines that couldn't be parsed, create a fallback
        if len(valid_lines) == 0:
            logger.warning("Could not extract any valid Python code. Returning fallback.")
            return "import pandas as pd\nresult = {\"type\": \"dataframe\", \"value\": pd.DataFrame()}"
            
        return '\n'.join(valid_lines)
    
    def is_openai(self) -> bool:
        """
        Check if this is an OpenAI LLM
        
        Returns:
            False, as this is a Groq LLM
        """
        return False 