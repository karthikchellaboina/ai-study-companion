import os

from dotenv import load_dotenv
from langfuse import get_client


# Load .env
load_dotenv()


def get_langfuse_client():
    """
    Return the shared Langfuse client.

    Langfuse reads:
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_BASE_URL
    """

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return None

    return get_client()


def is_langfuse_enabled():
    """
    Check whether Langfuse credentials are configured.
    """

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    return bool(public_key and secret_key)


def check_langfuse_connection():
    """
    Verify that Langfuse authentication is working.
    """

    if not is_langfuse_enabled():
        return {
            "enabled": False,
            "authenticated": False,
            "message": "Langfuse credentials are not configured.",
        }

    try:
        langfuse = get_langfuse_client()

        if langfuse is None:
            return {
                "enabled": False,
                "authenticated": False,
                "message": "Langfuse client could not be created.",
            }

        authenticated = langfuse.auth_check()

        if authenticated:
            return {
                "enabled": True,
                "authenticated": True,
                "message": "Langfuse is authenticated and ready.",
            }

        return {
            "enabled": True,
            "authenticated": False,
            "message": "Langfuse authentication failed.",
        }

    except Exception as e:
        return {
            "enabled": True,
            "authenticated": False,
            "message": f"Langfuse connection error: {e}",
        }


def create_test_trace():
    """
    Create a simple Langfuse test trace.

    The trace ID is captured while the observation
    context is still active.
    """

    langfuse = get_langfuse_client()

    if langfuse is None:
        return {
            "success": False,
            "message": "Langfuse is not configured.",
        }

    try:
        trace_id = None

        with langfuse.start_as_current_observation(
            as_type="span",
            name="ai-study-companion-test",
            input={
                "message": "Testing Langfuse tracing"
            },
        ) as span:

            # Capture trace ID INSIDE the active context
            trace_id = langfuse.get_current_trace_id()

            span.update(
                output={
                    "status": "success",
                    "message": "AI Study Companion Langfuse test trace",
                }
            )

        # Send trace data to Langfuse
        langfuse.flush()

        return {
            "success": True,
            "trace_id": trace_id,
            "message": "Test trace created successfully.",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to create test trace: {e}",
        }


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("🔎 LANGFUSE CONNECTION TEST")
    print("=" * 60)
    print()

    # Check authentication
    result = check_langfuse_connection()

    print(f"Enabled: {result['enabled']}")
    print(f"Authenticated: {result['authenticated']}")
    print(f"Message: {result['message']}")
    print()

    # Create test trace
    if result["authenticated"]:

        print("Creating test trace...")

        trace_result = create_test_trace()

        if trace_result["success"]:

            print("✅ Test trace created successfully.")

            if trace_result.get("trace_id"):
                print(f"Trace ID: {trace_result['trace_id']}")

        else:

            print("❌ Test trace failed.")
            print(trace_result["message"])

    print()
    print("=" * 60)