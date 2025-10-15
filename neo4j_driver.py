from neo4j import GraphDatabase

URI = "neo4j+ssc://bd07fa08.databases.neo4j.io"
AUTH = ("neo4j", "5d3Jlb3abtyNANIDQcvJERS9ndXdtL8hFw5oiSZrWew")


driver = GraphDatabase.driver(URI, auth=AUTH)