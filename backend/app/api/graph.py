"""
Graph-related API routes
Uses project context mechanism with server-side persistent state
"""

import os
from flask import request, jsonify

from . import graph_bp
from ..config import Config
from ..core.workbench_session import WorkbenchSession
from ..services.graph_builder import GraphBuilderService
from ..utils.logger import get_logger
from ..models.task import TaskManager
from ..models.project import ProjectManager, ProjectStatus

# Get logger
logger = get_logger('mirofish.api')


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if not filename or '.' not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    return ext in Config.ALLOWED_EXTENSIONS


# ============== Project Management API ==============

@graph_bp.route('/project/<project_id>', methods=['GET'])
def get_project(project_id: str):
    """
    Get project details
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": f"Project not found: {project_id}"
        }), 404

    return jsonify({
        "success": True,
        "data": project.to_dict()
    })


@graph_bp.route('/project/list', methods=['GET'])
def list_projects():
    """
    List all projects
    """
    limit = request.args.get('limit', 50, type=int)
    projects = ProjectManager.list_projects(limit=limit)
    
    return jsonify({
        "success": True,
        "data": [p.to_dict() for p in projects],
        "count": len(projects)
    })


@graph_bp.route('/project/<project_id>', methods=['DELETE'])
def delete_project(project_id: str):
    """
    Delete project
    """
    success = ProjectManager.delete_project(project_id)
    
    if not success:
        return jsonify({
            "success": False,
            "error": f"Project not found or deletion failed: {project_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "message": f"Project deleted: {project_id}"
    })


@graph_bp.route('/project/<project_id>/reset', methods=['POST'])
def reset_project(project_id: str):
    """
    Reset project state (for rebuilding the graph)
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": f"Project not found: {project_id}"
        }), 404

    # Reset to ontology-generated state
    if project.ontology:
        project.status = ProjectStatus.ONTOLOGY_GENERATED
    else:
        project.status = ProjectStatus.CREATED
    
    project.graph_id = None
    project.graph_build_task_id = None
    project.error = None
    ProjectManager.save_project(project)
    
    return jsonify({
        "success": True,
        "message": f"Project reset: {project_id}",
        "data": project.to_dict()
    })


# ============== API 1: Upload Files and Generate Ontology ==============

@graph_bp.route('/ontology/generate', methods=['POST'])
def generate_ontology():
    """API 1: Upload files and analyze to generate ontology definition."""
    try:
        logger.info("=== Starting ontology generation ===")

        simulation_requirement = request.form.get('simulation_requirement', '')
        project_name = request.form.get('project_name', 'Unnamed Project')
        additional_context = request.form.get('additional_context', '')
        uploaded_files = request.files.getlist('files')

        session = WorkbenchSession.open(metadata={"entrypoint": "api.graph.generate_ontology"})
        result = session.generate_ontology(
            simulation_requirement=simulation_requirement,
            uploaded_files=uploaded_files,
            project_name=project_name,
            additional_context=additional_context if additional_context else None,
        )

        return jsonify({
            "success": True,
            "data": result
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        logger.error(f"Failed to generate ontology: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============== API 2: Build Graph ==============

@graph_bp.route('/build', methods=['POST'])
def build_graph():
    """API 2: Build graph based on project_id."""
    try:
        logger.info("=== Starting graph build ===")

        data = request.get_json() or {}
        project_id = data.get('project_id')
        if not project_id:
            return jsonify({
                "success": False,
                "error": "Please provide project_id"
            }), 400

        session = WorkbenchSession.open(project_id=project_id, metadata={"entrypoint": "api.graph.build_graph"})
        result = session.start_graph_build(
            project_id=project_id,
            graph_name=data.get('graph_name'),
            chunk_size=data.get('chunk_size'),
            chunk_overlap=data.get('chunk_overlap'),
            force=data.get('force', False),
        )

        return jsonify({
            "success": True,
            "data": result
        })

    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        logger.error(f"Failed to start graph build: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============== Task Query API ==============

@graph_bp.route('/task/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """
    Query task status
    """
    task = TaskManager().get_task(task_id)
    
    if not task:
        return jsonify({
            "success": False,
            "error": f"Task not found: {task_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "data": task.to_dict()
    })


@graph_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """
    List all tasks
    """
    tasks = TaskManager().list_tasks()
    
    return jsonify({
        "success": True,
        "data": [t.to_dict() for t in tasks],
        "count": len(tasks)
    })


# ============== Graph Data API ==============

@graph_bp.route('/data/<graph_id>', methods=['GET'])
def get_graph_data(graph_id: str):
    """
    Get graph data (nodes and edges)
    """
    try:
        builder = GraphBuilderService()
        graph_data = builder.get_graph_data(graph_id)
        
        return jsonify({
            "success": True,
            "data": graph_data
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@graph_bp.route('/delete/<graph_id>', methods=['DELETE'])
def delete_graph(graph_id: str):
    """
    Delete graph
    """
    try:
        builder = GraphBuilderService()
        builder.delete_graph(graph_id)

        return jsonify({
            "success": True,
            "message": f"Graph deleted: {graph_id}"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
