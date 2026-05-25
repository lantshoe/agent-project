from langchain_core.tools import BaseTool
from pydantic import BaseModel

from agents.core.logger import get_logger

logger = get_logger("arg_validator")


def get_tool_schema(tool: BaseTool) -> dict:
    """
   Extracts field definitions from a tool's schema.
   Returns: {field_name: {"required": bool, "type": type, "description": str}}
   """
    schema = getattr(tool, "args_schema", None)
    if not schema:
        return {}

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    fields = {}
    required_fields = schema.get('required') or []
    for name, detail in (schema.get('properties') or {}).items():
        fields[name] = {
            "required": name in required_fields,
            "type": detail.get('type'),
            "description": detail.get('description')
        }
    return fields

def validate_and_fix_args(tool:BaseTool,args:dict) -> tuple[dict, list[str]]:
    """
       Systematically validates and fixes tool arguments.
       Applies three fixes:
         1. Strip unknown args
         2. Type coerce wrong types
         3. Report missing required args
       Returns: (fixed_args, list_of_warnings)
       """
    schema = get_tool_schema(tool)
    if not schema:
        return args, []

    warnings = []
    fixed = {}
    fixed = _handle_unknown(tool, args, schema, warnings, fixed)
    fixed = _correct_type(tool, args, schema, warnings, fixed)
    _report_missing_args(tool, args, schema, warnings, fixed)
    if warnings:
        logger.debug(f"[{tool.name}] Arg fixes: {warnings}")
    return fixed, warnings



def _handle_unknown(tool:BaseTool, args:dict, schema:dict, warnings, fixed):
    for key, value in args.items():
        if key not in schema:
            warnings.append(f"Stripped unknown arg '{key}'")
            logger.warning(f"[{tool.name}] Stripped unknown arg: '{key}={value}'")
        else:
            fixed[key] = value
    return fixed


def _correct_type(tool:BaseTool, args:dict, schema:dict,warnings,fixed ):
    for key, value in fixed.items():
        expected_type = schema[key]["type"]
        if expected_type is None:
            continue
        try:
            # Handle common mismatches
            if expected_type == "string" and not isinstance(value, str):
                fixed[key] = str(value)
                warnings.append(f"Coerced '{key}' to str")

            elif expected_type == "integer" and isinstance(value, float):
                fixed[key] = int(value)
                warnings.append(f"Coerced '{key}' to int")

            elif expected_type == "boolean" and isinstance(value, str):
                fixed[key] = value.lower() in {"true", "1", "yes"}
                warnings.append(f"Coerced '{key}' to bool")

            elif expected_type == "array" and isinstance(value, str):
                import json
                fixed[key] = json.loads(value)
                warnings.append(f"Coerced '{key}' from str to list")

        except Exception as e:
            warnings.append(f"Could not coerce '{key}': {e}")
    return fixed

def _report_missing_args(tool:BaseTool, args:dict, schema:dict, warnings:list, fixed):
    for name, field in schema.items():
        if field["required"] and name not in fixed:
            warnings.append(f"Missing required arg '{name}' ({field['description']})")
            logger.error(f"[{tool.name}] Missing required arg: '{name}'")
