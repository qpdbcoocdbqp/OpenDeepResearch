"""
Open Deep Research Client Library

This module provides a Python client for interacting with the Open Deep Research API.
The client supports both synchronous and asynchronous operations with comprehensive
error handling and usage examples.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, Union
import requests
import aiohttp
from dataclasses import dataclass
from enum import Enum


class ResearchStatus(Enum):
    """Research operation status."""
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class SearchAPI(Enum):
    """Available search API providers."""
    TAVILY = "tavily"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"


@dataclass
class ResearchResult:
    """Research operation result."""
    final_report: str
    research_brief: str
    status: ResearchStatus
    execution_time: float
    token_usage: Optional[Dict[str, int]] = None


@dataclass
class Configuration:
    """Research configuration settings."""
    summarization_model: str = "google_genai:gemini-2.5-flash-lite"
    research_model: str = "google_genai:gemini-2.5-flash-lite"
    compression_model: str = "google_genai:gemini-2.5-flash-lite"
    final_report_model: str = "google_genai:gemini-2.5-flash-lite"
    search_api: SearchAPI = SearchAPI.TAVILY
    max_concurrent_research_units: int = 5
    max_researcher_iterations: int = 6
    allow_clarification: bool = True


class OpenDeepResearchError(Exception):
    """Base exception for Open Deep Research client errors."""
    pass


class NetworkError(OpenDeepResearchError):
    """Network-related errors."""
    pass


class APIError(OpenDeepResearchError):
    """API-related errors."""
    def __init__(self, message: str, status_code: int, detail: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class TimeoutError(OpenDeepResearchError):
    """Timeout-related errors."""
    pass


class ValidationError(OpenDeepResearchError):
    """Query validation errors."""
    pass


class OpenDeepResearchClient:
    """
    Client for interacting with the Open Deep Research API.
    
    This client provides both synchronous and asynchronous methods for conducting
    research operations, managing configuration, and handling errors gracefully.
    
    Args:
        base_url: Base URL of the Open Deep Research API server
        timeout: Default timeout for requests in seconds
        max_retries: Maximum number of retry attempts for failed requests
        retry_delay: Delay between retry attempts in seconds
    
    Example:
        >>> client = OpenDeepResearchClient("http://localhost:8000")
        >>> result = client.research("What are the latest trends in AI?")
        >>> print(result.final_report)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 600,  # 10 minutes default
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            NetworkError: For network-related issues
            APIError: For API-related errors
            TimeoutError: For timeout issues
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
                
                # Handle successful responses
                if response.status_code == 200:
                    return response.json()
                
                # Handle API errors
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', f'HTTP {response.status_code}')
                    error_detail = error_data.get('detail')
                except (json.JSONDecodeError, KeyError):
                    error_message = f'HTTP {response.status_code}: {response.text}'
                    error_detail = None
                
                # Handle specific error types
                if response.status_code == 400:
                    raise ValidationError(f"Validation error: {error_message}")
                elif response.status_code == 401:
                    raise APIError(f"Authentication failed: {error_message}", response.status_code, error_detail)
                elif response.status_code == 408:
                    raise TimeoutError(f"Request timeout: {error_message}")
                elif response.status_code == 429:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    raise APIError(f"Rate limit exceeded: {error_message}", response.status_code, error_detail)
                elif response.status_code >= 500:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                        continue
                    raise APIError(f"Server error: {error_message}", response.status_code, error_detail)
                else:
                    raise APIError(f"API error: {error_message}", response.status_code, error_detail)
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                raise TimeoutError(f"Request timed out after {self.timeout} seconds")
            
            except requests.exceptions.ConnectionError as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                raise NetworkError(f"Connection error: {str(e)}")
            
            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error: {str(e)}")
        
        raise NetworkError("Max retries exceeded")
    
    def research(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        allow_clarification: bool = True
    ) -> ResearchResult:
        """
        Execute a research query synchronously.
        
        Args:
            query: Research question or topic to investigate
            config: Configuration overrides for the research process
            allow_clarification: Whether to allow clarifying questions
            
        Returns:
            ResearchResult containing the final report and metadata
            
        Raises:
            ValidationError: If the query is invalid
            APIError: If the API returns an error
            TimeoutError: If the request times out
            NetworkError: If there are network issues
            
        Example:
            >>> client = OpenDeepResearchClient()
            >>> result = client.research("What are the benefits of renewable energy?")
            >>> print(f"Research completed in {result.execution_time:.1f} seconds")
            >>> print(result.final_report)
        """
        request_data = {
            "query": query,
            "config": config or {},
            "allow_clarification": allow_clarification
        }
        
        response_data = self._make_request("POST", "/research", data=request_data)
        
        return ResearchResult(
            final_report=response_data["final_report"],
            research_brief=response_data["research_brief"],
            status=ResearchStatus(response_data["status"]),
            execution_time=response_data["execution_time"],
            token_usage=response_data.get("token_usage")
        )
    
    async def research_async(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        allow_clarification: bool = True
    ) -> ResearchResult:
        """
        Execute a research query asynchronously.
        
        Args:
            query: Research question or topic to investigate
            config: Configuration overrides for the research process
            allow_clarification: Whether to allow clarifying questions
            
        Returns:
            ResearchResult containing the final report and metadata
            
        Raises:
            ValidationError: If the query is invalid
            APIError: If the API returns an error
            TimeoutError: If the request times out
            NetworkError: If there are network issues
            
        Example:
            >>> client = OpenDeepResearchClient()
            >>> result = await client.research_async("How does machine learning work?")
            >>> print(result.final_report)
        """
        request_data = {
            "query": query,
            "config": config or {},
            "allow_clarification": allow_clarification
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            for attempt in range(self.max_retries + 1):
                try:
                    async with session.post(
                        f"{self.base_url}/research",
                        json=request_data,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        
                        if response.status == 200:
                            response_data = await response.json()
                            return ResearchResult(
                                final_report=response_data["final_report"],
                                research_brief=response_data["research_brief"],
                                status=ResearchStatus(response_data["status"]),
                                execution_time=response_data["execution_time"],
                                token_usage=response_data.get("token_usage")
                            )
                        
                        # Handle errors
                        try:
                            error_data = await response.json()
                            error_message = error_data.get('error', f'HTTP {response.status}')
                            error_detail = error_data.get('detail')
                        except (json.JSONDecodeError, KeyError):
                            error_message = f'HTTP {response.status}'
                            error_detail = None
                        
                        if response.status == 400:
                            raise ValidationError(f"Validation error: {error_message}")
                        elif response.status == 401:
                            raise APIError(f"Authentication failed: {error_message}", response.status, error_detail)
                        elif response.status == 408:
                            raise TimeoutError(f"Request timeout: {error_message}")
                        elif response.status == 429:
                            if attempt < self.max_retries:
                                await asyncio.sleep(self.retry_delay * (2 ** attempt))
                                continue
                            raise APIError(f"Rate limit exceeded: {error_message}", response.status, error_detail)
                        elif response.status >= 500:
                            if attempt < self.max_retries:
                                await asyncio.sleep(self.retry_delay)
                                continue
                            raise APIError(f"Server error: {error_message}", response.status, error_detail)
                        else:
                            raise APIError(f"API error: {error_message}", response.status, error_detail)
                
                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    raise TimeoutError(f"Request timed out after {self.timeout} seconds")
                
                except aiohttp.ClientError as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    raise NetworkError(f"Network error: {str(e)}")
        
        raise NetworkError("Max retries exceeded")
    
    def get_configuration(self) -> Configuration:
        """
        Get current server configuration.
        
        Returns:
            Configuration object with current settings
            
        Example:
            >>> client = OpenDeepResearchClient()
            >>> config = client.get_configuration()
            >>> print(f"Current research model: {config.research_model}")
        """
        response_data = self._make_request("GET", "/config")
        
        return Configuration(
            summarization_model=response_data["summarization_model"],
            research_model=response_data["research_model"],
            compression_model=response_data["compression_model"],
            final_report_model=response_data["final_report_model"],
            search_api=SearchAPI(response_data["search_api"]),
            max_concurrent_research_units=response_data["max_concurrent_research_units"],
            max_researcher_iterations=response_data["max_researcher_iterations"],
            allow_clarification=response_data["allow_clarification"]
        )
    
    def update_configuration(self, config: Configuration) -> Configuration:
        """
        Update server configuration.
        
        Args:
            config: New configuration settings
            
        Returns:
            Updated configuration
            
        Example:
            >>> client = OpenDeepResearchClient()
            >>> config = client.get_configuration()
            >>> config.max_concurrent_research_units = 3
            >>> updated_config = client.update_configuration(config)
        """
        config_data = {
            "summarization_model": config.summarization_model,
            "research_model": config.research_model,
            "compression_model": config.compression_model,
            "final_report_model": config.final_report_model,
            "search_api": config.search_api.value,
            "max_concurrent_research_units": config.max_concurrent_research_units,
            "max_researcher_iterations": config.max_researcher_iterations,
            "allow_clarification": config.allow_clarification
        }
        
        response_data = self._make_request("POST", "/config", data=config_data)
        
        return Configuration(
            summarization_model=response_data["summarization_model"],
            research_model=response_data["research_model"],
            compression_model=response_data["compression_model"],
            final_report_model=response_data["final_report_model"],
            search_api=SearchAPI(response_data["search_api"]),
            max_concurrent_research_units=response_data["max_concurrent_research_units"],
            max_researcher_iterations=response_data["max_researcher_iterations"],
            allow_clarification=response_data["allow_clarification"]
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check server health status.
        
        Returns:
            Health status information
            
        Example:
            >>> client = OpenDeepResearchClient()
            >>> health = client.health_check()
            >>> print(f"Server status: {health['status']}")
        """
        return self._make_request("GET", "/health")
    
    def validate_query(
        self,
        query: str,
        allow_clarification: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a research query without executing it.
        
        Args:
            query: Research query to validate
            allow_clarification: Whether clarification is allowed
            
        Returns:
            Validation results including complexity estimate
            
        Example:
            >>> client = OpenDeepResearchClient()
            >>> validation = client.validate_query("What is quantum computing?")
            >>> print(f"Query complexity: {validation['estimated_complexity']}")
        """
        request_data = {
            "query": query,
            "allow_clarification": allow_clarification
        }
        
        return self._make_request("POST", "/validate-query", data=request_data)
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Usage Examples and Demonstrations

def example_basic_research():
    """
    Example 1: Basic research query
    
    This example demonstrates the simplest way to use the client
    to conduct a research operation.
    """
    print("=== Example 1: Basic Research Query ===")
    
    # Create client instance
    client = OpenDeepResearchClient("http://localhost:8000")
    
    try:
        # Conduct research
        result = client.research("What are the environmental benefits of electric vehicles?")
        
        print(f"Research completed in {result.execution_time:.1f} seconds")
        print(f"Status: {result.status.value}")
        print("\n--- Research Brief ---")
        print(result.research_brief[:200] + "..." if len(result.research_brief) > 200 else result.research_brief)
        print("\n--- Final Report (excerpt) ---")
        print(result.final_report[:500] + "..." if len(result.final_report) > 500 else result.final_report)
        
        if result.token_usage:
            print(f"\nToken usage: {result.token_usage}")
            
    except ValidationError as e:
        print(f"Query validation failed: {e}")
    except APIError as e:
        print(f"API error (status {e.status_code}): {e}")
    except TimeoutError as e:
        print(f"Request timed out: {e}")
    except NetworkError as e:
        print(f"Network error: {e}")
    finally:
        client.close()


def example_custom_configuration():
    """
    Example 2: Research with custom configuration
    
    This example shows how to customize the research process
    by modifying configuration parameters.
    """
    print("\n=== Example 2: Custom Configuration ===")
    
    with OpenDeepResearchClient("http://localhost:8000") as client:
        try:
            # Get current configuration
            config = client.get_configuration()
            print(f"Current research model: {config.research_model}")
            print(f"Current search API: {config.search_api.value}")
            
            # Conduct research with custom config overrides
            custom_config = {
                "max_concurrent_research_units": 3,
                "max_researcher_iterations": 4
            }
            
            result = client.research(
                query="Compare different approaches to renewable energy storage",
                config=custom_config,
                allow_clarification=False
            )
            
            print(f"\nResearch completed with custom config in {result.execution_time:.1f} seconds")
            print("Final report generated successfully!")
            
        except Exception as e:
            print(f"Error: {e}")


async def example_async_research():
    """
    Example 3: Asynchronous research operations
    
    This example demonstrates how to use the async client
    for non-blocking research operations.
    """
    print("\n=== Example 3: Async Research ===")
    
    client = OpenDeepResearchClient("http://localhost:8000")
    
    try:
        # Multiple concurrent research queries
        queries = [
            "What are the latest developments in quantum computing?",
            "How does artificial intelligence impact healthcare?",
            "What are the challenges of space exploration?"
        ]
        
        # Execute all queries concurrently
        tasks = [
            client.research_async(query, allow_clarification=False)
            for query in queries
        ]
        
        print("Starting concurrent research operations...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        print(f"All research completed in {total_time:.1f} seconds")
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Query {i+1} failed: {result}")
            else:
                print(f"Query {i+1}: {result.status.value} ({result.execution_time:.1f}s)")
                
    except Exception as e:
        print(f"Error: {e}")


def example_error_handling():
    """
    Example 4: Comprehensive error handling
    
    This example demonstrates how to handle different types
    of errors that may occur during API interactions.
    """
    print("\n=== Example 4: Error Handling ===")
    
    client = OpenDeepResearchClient("http://localhost:8000", timeout=30)
    
    # Test cases for different error scenarios
    test_cases = [
        ("", "Empty query test"),
        ("AI", "Too short query test"),
        ("What is " + "very " * 2000 + "long query?", "Too long query test"),
        ("What are the latest trends in renewable energy?", "Valid query test")
    ]
    
    for query, description in test_cases:
        print(f"\n--- {description} ---")
        try:
            # First validate the query
            validation = client.validate_query(query)
            print(f"Query validation: {validation}")
            
            # Then attempt research if validation passes
            if validation.get('valid', False):
                result = client.research(query)
                print(f"Research successful: {result.status.value}")
            
        except ValidationError as e:
            print(f"Validation error: {e}")
        except APIError as e:
            print(f"API error (HTTP {e.status_code}): {e}")
            if e.detail:
                print(f"Details: {e.detail}")
        except TimeoutError as e:
            print(f"Timeout error: {e}")
        except NetworkError as e:
            print(f"Network error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    client.close()


def example_configuration_management():
    """
    Example 5: Configuration management
    
    This example shows how to retrieve, modify, and update
    server configuration settings.
    """
    print("\n=== Example 5: Configuration Management ===")
    
    with OpenDeepResearchClient("http://localhost:8000") as client:
        try:
            # Get current configuration
            print("Current configuration:")
            config = client.get_configuration()
            print(f"  Research model: {config.research_model}")
            print(f"  Search API: {config.search_api.value}")
            print(f"  Max concurrent units: {config.max_concurrent_research_units}")
            print(f"  Max iterations: {config.max_researcher_iterations}")
            print(f"  Allow clarification: {config.allow_clarification}")
            
            # Modify configuration
            print("\nUpdating configuration...")
            config.max_concurrent_research_units = 3
            config.max_researcher_iterations = 4
            config.search_api = SearchAPI.OPENAI
            
            # Update server configuration
            updated_config = client.update_configuration(config)
            print("Configuration updated successfully!")
            print(f"  New max concurrent units: {updated_config.max_concurrent_research_units}")
            print(f"  New search API: {updated_config.search_api.value}")
            
        except Exception as e:
            print(f"Configuration error: {e}")


def example_health_monitoring():
    """
    Example 6: Health monitoring and status checks
    
    This example demonstrates how to monitor server health
    and check system status.
    """
    print("\n=== Example 6: Health Monitoring ===")
    
    client = OpenDeepResearchClient("http://localhost:8000")
    
    try:
        # Check server health
        health = client.health_check()
        print(f"Server health: {health}")
        
        # Get detailed status (if available)
        try:
            status = client._make_request("GET", "/status")
            print(f"Detailed status: {status}")
        except APIError:
            print("Detailed status endpoint not available")
            
    except NetworkError as e:
        print(f"Cannot connect to server: {e}")
    except Exception as e:
        print(f"Health check failed: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    """
    Run all usage examples to demonstrate client functionality.
    
    Make sure the Open Deep Research server is running at http://localhost:8000
    before running these examples.
    """
    print("Open Deep Research Client - Usage Examples")
    print("=" * 50)
    
    # Run synchronous examples
    example_basic_research()
    example_custom_configuration()
    example_error_handling()
    example_configuration_management()
    example_health_monitoring()
    
    # Run async example
    print("\nRunning async example...")
    asyncio.run(example_async_research())
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo use the client in your own code:")
    print("1. Import: from client import OpenDeepResearchClient")
    print("2. Create: client = OpenDeepResearchClient('http://localhost:8000')")
    print("3. Research: result = client.research('Your research query here')")
    print("4. Use result: print(result.final_report)")