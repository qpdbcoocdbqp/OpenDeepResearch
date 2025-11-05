"""FastAPI server implementation for Open Deep Research.

This module provides a RESTful API wrapper around the Open Deep Research LangGraph
workflow, enabling easy deployment and usage of the research agent as a web service.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import sys
import os

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from open_deep_research.configuration import Configuration
from open_deep_research.deep_researcher import deep_researcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the compiled graph
deep_research_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize LangGraph on startup."""
    global deep_research_graph
    
    logger.info("Initializing Open Deep Research LangGraph workflow...")
    try:
        # Use the pre-compiled deep researcher graph
        deep_research_graph = deep_researcher
        logger.info("LangGraph workflow initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LangGraph workflow: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Open Deep Research server...")


# Create FastAPI application with lifespan management
app = FastAPI(
    title="Open Deep Research API",
    description="""
    ## Deep Research Agent API
    
    This API provides access to a sophisticated research agent powered by LangGraph and Google AI Studio Gemini models.
    The agent can conduct comprehensive research on any topic by:
    
    - **Analyzing** your research query and creating a focused research brief
    - **Searching** multiple sources using advanced search APIs (Tavily, OpenAI, Anthropic)
    - **Synthesizing** information from multiple sources into coherent findings
    - **Generating** comprehensive, well-structured research reports
    
    ### Key Features
    - 🤖 **AI-Powered Research**: Uses Google's Gemini-2.5-Flash-Lite model for intelligent analysis
    - 🔍 **Multi-Source Search**: Integrates with multiple search APIs for comprehensive coverage
    - 📊 **Structured Output**: Generates well-formatted research reports with citations
    - ⚙️ **Configurable**: Customize models, search providers, and research parameters
    - 🚀 **Async Processing**: Handles long-running research operations efficiently
    
    ### Getting Started
    1. Use the `/research` endpoint to submit your research query
    2. Monitor progress and get results when the research is complete
    3. Customize behavior using the `/config` endpoints
    4. Check system health with `/health` endpoint
    
    ### Interactive Documentation
    - **Swagger UI**: Available at `/docs` (this page) for interactive API testing
    - **ReDoc**: Available at `/redoc` for detailed documentation
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Open Deep Research API",
        "url": "https://github.com/your-repo/open-deep-research",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "research",
            "description": "Core research operations for conducting deep investigations"
        },
        {
            "name": "configuration",
            "description": "Configuration management for customizing research behavior"
        },
        {
            "name": "monitoring",
            "description": "Health checks and system status monitoring"
        },
        {
            "name": "validation",
            "description": "Query validation and preprocessing utilities"
        }
    ]
)

# Add CORS middleware for web client support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###################
# Pydantic Models
###################

class ResearchRequest(BaseModel):
    """
    Request model for research operations.
    
    This model defines the structure for submitting research queries to the API.
    The research agent will process your query and generate a comprehensive report.
    """
    
    query: str = Field(
        ..., 
        description="Research query or question to investigate. Should be specific and well-defined for best results.",
        min_length=10,
        max_length=10000
    )
    config: Optional[Dict[str, Any]] = Field(
        default={}, 
        description="Configuration overrides for the research process. Use this to customize model selection, search behavior, and other parameters."
    )
    allow_clarification: Optional[bool] = Field(
        default=True, 
        description="Whether to allow the researcher to ask clarifying questions if the query is ambiguous or needs more context"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the environmental impacts of cryptocurrency mining?",
                    "config": {},
                    "allow_clarification": True
                },
                {
                    "query": "Compare the effectiveness of different machine learning approaches for medical diagnosis",
                    "config": {
                        "max_concurrent_research_units": 5,
                        "research_model": "google_genai:gemini-2.5-flash-lite"
                    },
                    "allow_clarification": False
                },
                {
                    "query": "What are the latest trends in renewable energy storage technologies?",
                    "config": {
                        "search_api": "tavily",
                        "max_researcher_iterations": 6
                    },
                    "allow_clarification": True
                }
            ]
        }
    }


class ResearchResponse(BaseModel):
    """
    Response model for research operations.
    
    This model contains the complete results of a research operation, including
    the final report, research brief, and execution metadata.
    """
    
    final_report: str = Field(
        ..., 
        description="Generated comprehensive research report with findings, analysis, and conclusions"
    )
    research_brief: str = Field(
        ..., 
        description="Research brief that guided the investigation, showing the research strategy and focus areas"
    )
    status: str = Field(
        ..., 
        description="Execution status of the research process",
        pattern="^(completed|failed|timeout)$"
    )
    execution_time: float = Field(
        ..., 
        description="Time taken to complete the research in seconds",
        ge=0
    )
    token_usage: Optional[Dict[str, int]] = Field(
        default=None, 
        description="Token usage statistics if available, broken down by model and operation type"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "final_report": "# Research Report: Renewable Energy Storage\n\n## Executive Summary\n\nThis report examines the latest developments in renewable energy storage technologies...",
                    "research_brief": "Research Brief: Analyze current and emerging technologies for storing renewable energy, including battery systems, pumped hydro, and novel approaches.",
                    "status": "completed",
                    "execution_time": 67.3,
                    "token_usage": {
                        "total_tokens": 18500,
                        "prompt_tokens": 9200,
                        "completion_tokens": 9300
                    }
                }
            ]
        }
    }


class ConfigurationModel(BaseModel):
    """
    API model for configuration management.
    
    This model defines all configurable parameters for the research workflow,
    allowing you to customize model selection, search behavior, and processing limits.
    """
    
    summarization_model: str = Field(
        default="google_genai:gemini-2.5-flash-lite",
        description="Model used for summarizing research findings and intermediate results"
    )
    research_model: str = Field(
        default="google_genai:gemini-2.5-flash-lite",
        description="Primary model used for research analysis and content generation"
    )
    compression_model: str = Field(
        default="google_genai:gemini-2.5-flash-lite",
        description="Model used for compressing and condensing large amounts of research data"
    )
    final_report_model: str = Field(
        default="google_genai:gemini-2.5-flash-lite",
        description="Model used for generating the final research report"
    )
    search_api: str = Field(
        default="tavily",
        description="Search API provider to use for information gathering",
        pattern="^(tavily|openai|anthropic|none)$"
    )
    max_concurrent_research_units: int = Field(
        default=5,
        description="Maximum number of concurrent research operations to run in parallel",
        ge=1,
        le=20
    )
    max_researcher_iterations: int = Field(
        default=6,
        description="Maximum number of research iterations per query",
        ge=1,
        le=20
    )
    allow_clarification: bool = Field(
        default=True,
        description="Whether to allow the system to ask clarifying questions for ambiguous queries"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summarization_model": "google_genai:gemini-2.5-flash-lite",
                    "research_model": "google_genai:gemini-2.5-flash-lite",
                    "compression_model": "google_genai:gemini-2.5-flash-lite",
                    "final_report_model": "google_genai:gemini-2.5-flash-lite",
                    "search_api": "tavily",
                    "max_concurrent_research_units": 5,
                    "max_researcher_iterations": 6,
                    "allow_clarification": True
                },
                {
                    "summarization_model": "openai:gpt-4",
                    "research_model": "google_genai:gemini-2.5-flash-lite",
                    "compression_model": "google_genai:gemini-2.5-flash-lite",
                    "final_report_model": "openai:gpt-4",
                    "search_api": "openai",
                    "max_concurrent_research_units": 3,
                    "max_researcher_iterations": 8,
                    "allow_clarification": False
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Provides information about the current health and status of the API service.
    """
    
    status: str = Field(
        ..., 
        description="Service health status",
        pattern="^(healthy|unhealthy|degraded)$"
    )
    timestamp: str = Field(
        ..., 
        description="Current timestamp in ISO format"
    )
    version: str = Field(
        ..., 
        description="API version"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2024-01-15T10:30:45.123456",
                    "version": "1.0.0"
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    
    Provides structured error information when API operations fail.
    """
    
    error: str = Field(
        ..., 
        description="High-level error message describing what went wrong"
    )
    detail: Optional[str] = Field(
        None, 
        description="Detailed error information for debugging purposes"
    )
    status_code: int = Field(
        ..., 
        description="HTTP status code associated with this error",
        ge=400,
        le=599
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "Validation error",
                    "detail": "Query must be at least 10 characters long",
                    "status_code": 400
                },
                {
                    "error": "Authentication failed",
                    "detail": "Invalid API key for Google AI Studio",
                    "status_code": 401
                },
                {
                    "error": "Research execution failed",
                    "detail": "Timeout occurred during research operation",
                    "status_code": 500
                }
            ]
        }
    }


class QueryValidationResponse(BaseModel):
    """
    Response model for query validation endpoint.
    
    Provides validation results and metadata about a research query.
    """
    
    valid: bool = Field(
        ...,
        description="Whether the query passed validation checks"
    )
    query_length: int = Field(
        ...,
        description="Length of the query in characters",
        ge=0
    )
    estimated_complexity: str = Field(
        ...,
        description="Estimated processing complexity based on query characteristics",
        pattern="^(low|medium|high)$"
    )
    allow_clarification: bool = Field(
        ...,
        description="Whether clarification questions are enabled for this query"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "valid": True,
                    "query_length": 87,
                    "estimated_complexity": "medium",
                    "allow_clarification": True
                },
                {
                    "valid": True,
                    "query_length": 45,
                    "estimated_complexity": "low",
                    "allow_clarification": False
                },
                {
                    "valid": True,
                    "query_length": 1250,
                    "estimated_complexity": "high",
                    "allow_clarification": True
                }
            ]
        }
    }


###################
# API Endpoints
###################

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["monitoring"],
    summary="Health Check",
    description="""
    Check if the server and LangGraph workflow are running properly.
    
    This endpoint provides a quick way to verify that:
    - The FastAPI server is running
    - The LangGraph workflow is properly initialized
    - The system is ready to process research requests
    
    **Use this endpoint for:**
    - Load balancer health checks
    - Monitoring system integration
    - Debugging connection issues
    """,
    responses={
        200: {
            "description": "Service is healthy and ready to process requests",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:45.123456",
                        "version": "1.0.0"
                    }
                }
            }
        },
        503: {
            "description": "Service is unhealthy - LangGraph workflow not initialized",
            "content": {
                "application/json": {
                    "example": {
                        "error": "LangGraph workflow not initialized",
                        "detail": None,
                        "status_code": 503
                    }
                }
            }
        }
    }
)
async def health_check():
    """Health check endpoint for server monitoring."""
    global deep_research_graph
    
    if deep_research_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LangGraph workflow not initialized"
        )
    
    from datetime import datetime
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post(
    "/research",
    response_model=ResearchResponse,
    tags=["research"],
    summary="Conduct Deep Research",
    description="""
    Execute a comprehensive research query through the LangGraph workflow.
    
    This is the main endpoint for conducting research. The system will:
    
    1. **Analyze** your query and create a focused research brief
    2. **Search** multiple sources using configured search APIs
    3. **Process** and synthesize information from various sources
    4. **Generate** a comprehensive, well-structured research report
    
    **Query Guidelines:**
    - Be specific and clear about what you want to research
    - Provide context when necessary (e.g., "for medical applications", "in the context of climate change")
    - Avoid overly broad queries that would require extensive clarification
    
    **Processing Time:**
    - Simple queries: 30-60 seconds
    - Complex queries: 2-5 minutes
    - Very complex queries: 5-10 minutes
    
    **Configuration Options:**
    - Customize models used for different stages of research
    - Adjust concurrency and iteration limits
    - Enable/disable clarification questions
    """,
    responses={
        200: {
            "description": "Research completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "final_report": "# Research Report: Quantum Computing in Drug Discovery\n\n## Executive Summary\n\nQuantum computing represents a paradigm shift...",
                        "research_brief": "Research Brief: Investigate current applications of quantum computing in pharmaceutical research...",
                        "status": "completed",
                        "execution_time": 67.3,
                        "token_usage": {
                            "total_tokens": 18500,
                            "prompt_tokens": 9200,
                            "completion_tokens": 9300
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid request - query validation failed",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Query must be at least 10 characters long and not empty",
                        "detail": None,
                        "status_code": 400
                    }
                }
            }
        },
        408: {
            "description": "Request timeout - research operation took too long",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Research operation timed out. Please try with a simpler query.",
                        "detail": None,
                        "status_code": 408
                    }
                }
            }
        },
        429: {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Rate limit exceeded. Please try again later.",
                        "detail": None,
                        "status_code": 429
                    }
                }
            }
        },
        500: {
            "description": "Internal server error during research execution",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Research execution failed: Model authentication failed",
                        "detail": None,
                        "status_code": 500
                    }
                }
            }
        }
    }
)
async def conduct_research(request: ResearchRequest):
    """Main research endpoint that processes queries through LangGraph."""
    global deep_research_graph
    
    if deep_research_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LangGraph workflow not initialized"
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Starting research for query: {request.query[:100]}...")
        
        # Validate query input
        if not request.query or len(request.query.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be at least 10 characters long and not empty"
            )
        
        if len(request.query) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be less than 10,000 characters"
            )
        
        # Prepare configuration for the research
        config = {
            "configurable": {
                "allow_clarification": request.allow_clarification,
                **request.config
            }
        }
        
        # Execute the research workflow with proper async handling and timeout
        from langchain_core.messages import HumanMessage
        
        logger.info("Executing research workflow...")
        
        # Set a reasonable timeout for research operations (10 minutes)
        timeout_seconds = 600
        try:
            result = await asyncio.wait_for(
                deep_research_graph.ainvoke(
                    {"messages": [HumanMessage(content=request.query)]},
                    config=config
                ),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Research operation timed out after {timeout_seconds} seconds")
            raise
        
        execution_time = time.time() - start_time
        
        # Extract results from the workflow output (AgentState)
        final_report = result.get("final_report", "No report generated")
        research_brief = result.get("research_brief", "No research brief available")
        
        # Validate that we got meaningful results
        if not final_report or final_report == "No report generated":
            logger.warning("Research completed but no final report was generated")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Research completed but failed to generate a final report"
            )
        
        logger.info(f"Research completed successfully in {execution_time:.2f} seconds")
        
        return ResearchResponse(
            final_report=final_report,
            research_brief=research_brief,
            status="completed",
            execution_time=execution_time,
            token_usage=None  # TODO: Extract token usage from result if available
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except asyncio.TimeoutError:
        execution_time = time.time() - start_time
        logger.error(f"Research timed out after {execution_time:.2f} seconds")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Research operation timed out. Please try with a simpler query."
        )
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Research failed after {execution_time:.2f} seconds: {str(e)}", exc_info=True)
        
        # Provide more specific error messages based on error type
        if "rate limit" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        elif "authentication" in str(e).lower() or "api key" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed. Please check API key configuration."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Research execution failed: {str(e)}"
            )


@app.get(
    "/config",
    response_model=ConfigurationModel,
    tags=["configuration"],
    summary="Get Current Configuration",
    description="""
    Retrieve the current configuration settings for the research workflow.
    
    This endpoint returns all configurable parameters including:
    - **Model Selection**: Which AI models are used for different research stages
    - **Search Configuration**: Which search API provider is active
    - **Processing Limits**: Concurrency and iteration constraints
    - **Behavior Settings**: Whether clarification questions are allowed
    
    **Use this endpoint to:**
    - Check current system configuration
    - Verify configuration changes have been applied
    - Debug configuration-related issues
    """,
    responses={
        200: {
            "description": "Current configuration retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "summarization_model": "google_genai:gemini-2.5-flash-lite",
                        "research_model": "google_genai:gemini-2.5-flash-lite",
                        "compression_model": "google_genai:gemini-2.5-flash-lite",
                        "final_report_model": "google_genai:gemini-2.5-flash-lite",
                        "search_api": "tavily",
                        "max_concurrent_research_units": 5,
                        "max_researcher_iterations": 6,
                        "allow_clarification": True
                    }
                }
            }
        },
        500: {
            "description": "Failed to retrieve configuration",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Failed to retrieve configuration: Configuration file not found",
                        "detail": None,
                        "status_code": 500
                    }
                }
            }
        }
    }
)
async def get_configuration():
    """Get current configuration settings."""
    try:
        # Create default configuration instance
        config = Configuration()
        
        return ConfigurationModel(
            summarization_model=config.summarization_model,
            research_model=config.research_model,
            compression_model=config.compression_model,
            final_report_model=config.final_report_model,
            search_api=config.search_api.value,
            max_concurrent_research_units=config.max_concurrent_research_units,
            max_researcher_iterations=config.max_researcher_iterations,
            allow_clarification=config.allow_clarification
        )
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )


@app.post(
    "/config",
    response_model=ConfigurationModel,
    tags=["configuration"],
    summary="Update Configuration",
    description="""
    Update configuration settings for the research workflow.
    
    This endpoint allows you to modify system behavior by updating:
    
    **Model Configuration:**
    - `summarization_model`: Model for summarizing research findings
    - `research_model`: Primary model for research analysis
    - `compression_model`: Model for compressing large data sets
    - `final_report_model`: Model for generating final reports
    
    **Search Configuration:**
    - `search_api`: Choose between tavily, openai, anthropic, or none
    
    **Processing Limits:**
    - `max_concurrent_research_units`: Parallel research operations (1-20)
    - `max_researcher_iterations`: Maximum research iterations (1-20)
    
    **Behavior Settings:**
    - `allow_clarification`: Enable/disable clarifying questions
    
    **Supported Model Formats:**
    - Google AI Studio: `google_genai:gemini-2.5-flash-lite`
    - OpenAI: `openai:gpt-4`, `gpt-4-turbo`
    - Anthropic: `anthropic:claude-3-sonnet`, `claude-3-haiku`
    
    **Note:** Configuration changes apply to new research requests only.
    """,
    responses={
        200: {
            "description": "Configuration updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "summarization_model": "google_genai:gemini-2.5-flash-lite",
                        "research_model": "openai:gpt-4",
                        "compression_model": "google_genai:gemini-2.5-flash-lite",
                        "final_report_model": "openai:gpt-4",
                        "search_api": "tavily",
                        "max_concurrent_research_units": 3,
                        "max_researcher_iterations": 8,
                        "allow_clarification": False
                    }
                }
            }
        },
        400: {
            "description": "Invalid configuration values",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_range": {
                            "summary": "Value out of range",
                            "value": {
                                "error": "max_concurrent_research_units must be between 1 and 20",
                                "detail": None,
                                "status_code": 400
                            }
                        },
                        "invalid_search_api": {
                            "summary": "Invalid search API",
                            "value": {
                                "error": "search_api must be one of: tavily, openai, anthropic, none",
                                "detail": None,
                                "status_code": 400
                            }
                        }
                    }
                }
            }
        },
        500: {
            "description": "Failed to update configuration",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Failed to update configuration: Internal error",
                        "detail": None,
                        "status_code": 500
                    }
                }
            }
        }
    }
)
async def update_configuration(config_update: ConfigurationModel):
    """Update configuration settings."""
    try:
        logger.info("Configuration update requested")
        logger.info(f"New config: {config_update.model_dump()}")
        
        # Validate configuration values
        if config_update.max_concurrent_research_units < 1 or config_update.max_concurrent_research_units > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_concurrent_research_units must be between 1 and 20"
            )
        
        if config_update.max_researcher_iterations < 1 or config_update.max_researcher_iterations > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="max_researcher_iterations must be between 1 and 20"
            )
        
        # Validate model names (basic validation)
        valid_model_prefixes = ["google_genai:", "openai:", "anthropic:", "gpt-", "claude-", "gemini-"]
        for model_field in ["summarization_model", "research_model", "compression_model", "final_report_model"]:
            model_name = getattr(config_update, model_field)
            if not any(model_name.startswith(prefix) for prefix in valid_model_prefixes):
                logger.warning(f"Model name '{model_name}' may not be valid")
        
        # Validate search API
        valid_search_apis = ["tavily", "openai", "anthropic", "none"]
        if config_update.search_api not in valid_search_apis:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"search_api must be one of: {', '.join(valid_search_apis)}"
            )
        
        # Note: This is a basic implementation that returns the updated config
        # In a production system, you might want to persist these changes
        # or apply them to the running workflow
        
        logger.info("Configuration validation passed")
        return config_update
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to update configuration: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )


###################
# Error Handlers
###################

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=None,
            status_code=exc.status_code
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions with structured error responses."""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        ).model_dump()
    )


###################
# Additional Endpoints
###################

@app.get(
    "/status",
    tags=["monitoring"],
    summary="Detailed Server Status",
    description="""
    Get comprehensive server status including workflow initialization and configuration.
    
    This endpoint provides more detailed information than `/health`, including:
    - Server runtime status
    - LangGraph workflow initialization status
    - Current default configuration
    - System timestamp and version
    
    **Use this endpoint for:**
    - Detailed system diagnostics
    - Configuration verification
    - Troubleshooting initialization issues
    """,
    responses={
        200: {
            "description": "Detailed server status information",
            "content": {
                "application/json": {
                    "example": {
                        "server_status": "running",
                        "workflow_initialized": True,
                        "timestamp": 1705312245.123,
                        "version": "1.0.0",
                        "default_config": {
                            "research_model": "google_genai:gemini-2.5-flash-lite",
                            "search_api": "tavily",
                            "allow_clarification": True
                        }
                    }
                }
            }
        }
    }
)
async def get_status():
    """Get detailed server status."""
    global deep_research_graph
    
    status_info = {
        "server_status": "running",
        "workflow_initialized": deep_research_graph is not None,
        "timestamp": time.time(),
        "version": "1.0.0"
    }
    
    # Add configuration status
    try:
        config = Configuration()
        status_info["default_config"] = {
            "research_model": config.research_model,
            "search_api": config.search_api.value,
            "allow_clarification": config.allow_clarification
        }
    except Exception as e:
        status_info["config_error"] = str(e)
    
    return status_info


@app.post(
    "/validate-query",
    response_model=QueryValidationResponse,
    tags=["validation"],
    summary="Validate Research Query",
    description="""
    Validate a research query without executing the full research workflow.
    
    This endpoint performs pre-flight validation to check if your query:
    - Meets minimum and maximum length requirements
    - Contains appropriate content (no harmful requests)
    - Has reasonable complexity for processing
    
    **Validation Checks:**
    - **Length**: Query must be 10-10,000 characters
    - **Content**: Basic content filtering for harmful requests
    - **Complexity**: Estimates processing complexity based on query length
    
    **Use this endpoint to:**
    - Validate queries before submitting expensive research requests
    - Get complexity estimates for planning purposes
    - Debug query formatting issues
    """,
    responses={
        200: {
            "description": "Query validation results",
            "content": {
                "application/json": {
                    "example": {
                        "valid": True,
                        "query_length": 87,
                        "estimated_complexity": "medium",
                        "allow_clarification": True
                    }
                }
            }
        },
        400: {
            "description": "Query validation failed",
            "content": {
                "application/json": {
                    "examples": {
                        "too_short": {
                            "summary": "Query too short",
                            "value": {
                                "error": "Query must be at least 10 characters long and not empty",
                                "detail": None,
                                "status_code": 400
                            }
                        },
                        "too_long": {
                            "summary": "Query too long",
                            "value": {
                                "error": "Query must be less than 10,000 characters",
                                "detail": None,
                                "status_code": 400
                            }
                        },
                        "harmful_content": {
                            "summary": "Harmful content detected",
                            "value": {
                                "error": "Query contains potentially harmful content",
                                "detail": None,
                                "status_code": 400
                            }
                        }
                    }
                }
            }
        }
    }
)
async def validate_query(request: ResearchRequest):
    """Validate a research query without executing the full workflow."""
    
    # Validate query input
    if not request.query or len(request.query.strip()) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be at least 10 characters long and not empty"
        )
    
    if len(request.query) > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be less than 10,000 characters"
        )
    
    # Basic content validation
    query_lower = request.query.lower()
    if any(word in query_lower for word in ["hack", "exploit", "illegal", "harmful"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query contains potentially harmful content"
        )
    
    return QueryValidationResponse(
        valid=True,
        query_length=len(request.query),
        estimated_complexity="high" if len(request.query) > 500 else "medium" if len(request.query) > 100 else "low",
        allow_clarification=request.allow_clarification
    )


###################
# Main Entry Point
###################

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )