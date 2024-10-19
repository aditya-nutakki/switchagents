from anthropic import Anthropic
from openai import OpenAI

from utils import *
import os, json

from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

import torch


class BaseAgent():
    def __init__(self, bundle) -> None:
        self.id = bundle["id"]
        self.model = bundle["model"]
        self.tools = bundle["tools"]
        self.temperature = bundle["temperature"]
        
        if bundle["stream_def"]:
            self.stream_def, self.default_behaviour = bundle["stream_def"]
        else:
            self.stream_def, self.default_behaviour = None, None

        self.available_models = bundle["available_models"]
        self.base_url = bundle["base_url"]
        self.is_conditional_stream = True if self.stream_def else False
        self.lag_len = 4


    def _find_slave(self, model):
        idx = self.available_models.index(model)
        if idx == 0:
            return model
        return self.available_models[idx - 1]


    def process_prompt(self, role, content):
        return [{"role": role, "content": content}]


    def _get_latest_user_query(self, context):
        for message in context:
            if message["role"] == "user":
                query = message["content"] # rewritten until the very last user query
        return query


    def _remove_stream_defs(self, text_response):

        def process_tag_name(tag):
            return tag.replace("<","").replace(">", "")

        def extract_tag_content(text, tags):
            tags = [process_tag_name(tag) for tag in tags]
            pattern = r"<(" + "|".join(tags) + r")>(.*?)<\/\1>"
            matches = re.findall(pattern, text)
            x = [content[1].strip() for content in matches]
            return "\n".join(x)


        def extract_non_tag_content(text, whitelist_tags, blacklist_tags):
            blacklist_tags = [process_tag_name(tag) for tag in blacklist_tags]
            whitelist_tags = [process_tag_name(tag) for tag in whitelist_tags]
            
            for tag in blacklist_tags:
                pattern = r'(?:<' + tag + r'>.*?<\/(?:' + tag + ')>)'
                text = re.sub(pattern, '', text)

            pattern = r'</?(' + "|".join(whitelist_tags) + r')>'
            result = re.sub(pattern, '\n', text)
            result = "\n".join([res.strip() for res in result.split("\n")])
            return result

        stream_def, default_behaviour = self.stream_def, self.default_behaviour
        if stream_def:

            # when default behaviour is blacklist, i can only take in whitelist tag
            # when default behaviour is whitelist, i can only remove blacklist tag

            if default_behaviour == "blacklist":
                tags_to_care = []
                for tag in list(stream_def.keys()):
                    if stream_def[tag]["behaviour"] == "whitelist":
                        tags_to_care.append(tag)

                clean_text = extract_tag_content(text_response, tags = tags_to_care)
                return clean_text
            

            else:
                blacklist_tags, whitelist_tags = [], []
                tags_to_care = []
                for tag in list(stream_def.keys()):
                    if stream_def[tag]["behaviour"] == "whitelist":
                        tags_to_care.append(tag)
                        whitelist_tags.append(tag)
                    else:
                        blacklist_tags.append(tag)
                        
                clean_text = extract_non_tag_content(text_response, whitelist_tags = whitelist_tags, blacklist_tags = blacklist_tags)
                return clean_text

        else:
            print("No stream def, returning as is")
            return text_response



class BaseOpenAIAgent(BaseAgent):

    """
        It seems that the OpenAI client is increasingly growing as an API base for many other providers.
        They are beginning to use OpenAI's client with their own keys and base urls. Quick + easy deployment and it makes sense to do so. 
        This Base agent will have to cater to this problem as well. It is okay to write some more custom code on top of this.
    """

    def __init__(self, bundle):
        super().__init__(bundle = bundle)

        self.system = self.process_prompt(role = "system", content = bundle["system_prompt"])
        self.client = self._init_client(base_url = self.base_url, key = os.getenv(bundle["key_env_variable"]))

        if not self.tools:
            self.tools = None # OpenAI client wants tools to be 'None' and not '[]'

        print(f"Initializing id with {self.id}\n")
        print(f"Initializing model with {self.model}\n")
        print(f"Initializing system_prompt with {self.system}\n")
        print(f"Initializing tools with {self.tools}\n")
        print(f"Initializing temperature with {self.temperature}\n")
        print(f"Initializing stream_def with {self.stream_def}\n")
        print(f"Initializing available_models with {self.available_models}\n")
        print(f"Initializing base_url with {self.base_url}\n")
        print("___________________________________________")


    def _init_client(self, base_url, key):
        if base_url:
            client = OpenAI(base_url = base_url, api_key = key)
        else:
            client = OpenAI(api_key = key)
        
        return client
    

    # def get_embedding(self, text, model = "text-embedding-3-small"):
    def get_embedding(self, text):
        # "text" here is a string

        # text = text.replace("\n", " ").strip()
        # returns a list of 1536 for 'text-embedding-3-small' and a list of 3072 for 'text-embedding-3-large' 
        try:
            return self.client.embeddings.create(input = [text], model = self.model).data[0].embedding
        except Exception as e:
            print(f"failed due to {e} where text: {text}, {len(text), {type(text)}}")
            return []


    def _unconditional_stream(self, response, user_query):
        text_stream, json_stream = "", ""
        total_stream = ""

        for chunk in response:

            if chunk.choices[0].delta.content:
                chunk_content = chunk.choices[0].delta.content 
                text_stream += chunk_content
                total_stream += chunk_content
                yield chunk_content

            elif chunk.choices[0].delta.tool_calls:
                tool_calls = chunk.choices[0].delta.tool_calls
                if tool_calls[0].function.name:
                    func_name = tool_calls[0].function.name
                    json_stream = "" # introducing this here, because we are only going to be calling one function at a time

                elif tool_calls[0].function.arguments:
                    json_stream += tool_calls[0].function.arguments
                    total_stream += tool_calls[0].function.arguments

        if json_stream:
            # execute whatever function call
            json_stream = json.loads(json_stream)
            print(f"calling {func_name} with {json_stream} ...")
            # eval(func_name)(**json_response) 
    
            func_response = getattr(self, func_name)(user_query, **json_stream) # this function should also take care of processing calls again and should exist within the object
            for chunk in func_response:
                total_stream += chunk
                yield chunk


    def _conditional_stream(self, response, user_query):
        def process_stream(word_stream, current_behaviour):
            if current_behaviour == "blacklist":
                return ""
            
            elif current_behaviour == "record":
                # this happens only if you've already hit a record function;
                streams["record_stream"] += word_stream[:x]
                return ""
            
            else:
                # whitelist
                streams["allowed_stream"] += word_stream[:x]
                return word_stream[:x]


        def process_default_stream(word_stream, current_behaviour):
            if current_behaviour == "blacklist":
                # print("here4")
                return ""
            
            elif current_behaviour == "record":
                # print("here5")
                # record_stream += word_stream
                streams["record_stream"] += word_stream
                return ""
            
            else:
                # whitelist scenario
                streams["allowed_stream"] += word_stream
                return word_stream


        def process_remaining_stream(word_stream, current_behaviour):
            if current_behaviour == "blacklist":
                return ""
            
            elif current_behaviour == "record":
                # record_stream += word_stream[y:]
                streams["record_stream"] += word_stream[y:]
                return ""
            else:
                # allowed_stream += word_stream[y:]
                streams["allowed_stream"] += word_stream[y:]
                return word_stream[y:]

        # record_stream, allowed_stream, word_stream = "", "", ""
        streams = {"record_stream": "", "allowed_stream": ""}
        word_stream = ""
        current_behaviour = self.default_behaviour

        look_for_defs = list(self.stream_def.keys())
        found_closing_tag = False
        total_stream = []
        json_stream = ""

        for chunk in response:
            # if isinstance(chunk, anthropic.types.content_block_delta_event.ContentBlockDeltaEvent):
            # if chunk.type == "content_block_delta":
            if chunk.choices[0].delta.content:
                # if chunk.delta.type == "text_delta":
                chunk = chunk.choices[0].delta.content 
                total_stream.append(chunk) # present for debugging
                word_stream += chunk
                
                if len(word_stream.split(" ")) >= self.lag_len:
                    # print(f"looking for {look_for_defs}")
                    analysis = analyse_stream(word_stream, self.stream_def, look_for_defs = look_for_defs, current_behaviour = current_behaviour, found_closing_tag = found_closing_tag)
                    if analysis:
                        # you must check for current behaviour right here and then update later
                    
                        x, y, closing_tag, found_closing_tag, look_for_defs = process_analysis(analysis, self.stream_def, return_behaviour=False) # ensure return_behaviour is False
                        
                        yield process_stream(word_stream, current_behaviour = current_behaviour)
                        
                        # check if there's another opening tag within the same stream. This happens more often than not  
                        _remaining_word_stream = word_stream[y:]
                        
                        # it is important to call 'current_behaviour' here in order to have these changes only AFTER encountering that tag. Till the opening tag has been encountered, behaviour must be the previous state/default
                        current_behaviour = analysis["behaviour"] if analysis["behaviour"] else self.default_behaviour
                        if not found_closing_tag:
                            current_behaviour = self.default_behaviour

                        if _remaining_word_stream:
                            remaining_analysis = analyse_stream(_remaining_word_stream, stream_def = self.stream_def, look_for_defs = look_for_defs, current_behaviour = current_behaviour, found_closing_tag = found_closing_tag)

                            if remaining_analysis:
                                x, y, closing_tag, found_closing_tag, look_for_defs, current_behaviour = process_analysis(remaining_analysis, stream_def = self.stream_def, return_behaviour = True) # ensure return_behaviour is True

                                # opening tag

                                yield process_remaining_stream(word_stream=_remaining_word_stream, current_behaviour=current_behaviour)


                            else:
                                yield process_remaining_stream(word_stream, current_behaviour)

                
                    else:
                        yield process_default_stream(word_stream, current_behaviour = current_behaviour)

                    word_stream = ""

            elif chunk.choices[0].delta.tool_calls:
                tool_calls = chunk.choices[0].delta.tool_calls
                if tool_calls[0].function.name:
                    func_name = tool_calls[0].function.name
                    json_stream = "" # introducing this here, because we are only going to be calling one function at a time

                elif tool_calls[0].function.arguments:
                    json_stream += tool_calls[0].function.arguments
                    total_stream += tool_calls[0].function.arguments


        if word_stream:
            # final check for closing tag or whatever
            remaining_analysis = analyse_stream(word_stream, stream_def = self.stream_def, look_for_defs = look_for_defs, current_behaviour = current_behaviour, found_closing_tag = found_closing_tag)
            if remaining_analysis:
                x, y, closing_tag, found_closing_tag, look_for_defs, current_behaviour = process_analysis(remaining_analysis, stream_def = self.stream_def, return_behaviour = True) # ensure return_behaviour is True

                yield process_stream(word_stream, current_behaviour=current_behaviour)

                _remaining_word_stream = word_stream[y:]
                        
                # it is important to call 'current_behaviour' here in order to have these changes only AFTER encountering that tag. Till the opening tag has been encountered, behaviour must be the previous state/default
                current_behaviour = remaining_analysis["behaviour"] if remaining_analysis["behaviour"] else self.default_behaviour
                if not found_closing_tag:
                    current_behaviour = self.default_behaviour

                if _remaining_word_stream:
                    remaining_analysis = analyse_stream(_remaining_word_stream, stream_def = self.stream_def, look_for_defs = look_for_defs, current_behaviour = current_behaviour, found_closing_tag = found_closing_tag)

                    if remaining_analysis:
                        x, y, closing_tag, found_closing_tag, look_for_defs, current_behaviour = process_analysis(remaining_analysis, stream_def = self.stream_def, return_behaviour = True) # ensure return_behaviour is True

                        # opening tag
                        yield process_remaining_stream(_remaining_word_stream, current_behaviour)
                        
                    
                    else:
                        yield process_remaining_stream(word_stream, current_behaviour)
                    
                    word_stream = ""
            # print(allowed_stream)

            else:
                yield process_default_stream(word_stream, current_behaviour = current_behaviour)
                

        # print(total_stream)
        print(f"FINALLL RECORD STREAM: {streams['record_stream']}")

        if json_stream:

            print(f"JSON STREAM: {json_stream} with func_name {func_name}")
            json_stream = json.loads(json_stream)
            print("ABOUT TO CALL")
            func_response = getattr(self, func_name)(user_query, **json_stream)
            for chunk in func_response:
                total_stream += chunk
                yield chunk
    
        print(f"**************** {''.join(total_stream)}")
    

    def _call_once(self, context, system = "", model = "", temperature = None, tools = None, stream = False, use_slave = False):
        
        system = self.process_prompt(role = "system", content = system) if system else self.system
        model = model if model else self.model
        temperature = temperature if temperature else self.temperature
        tools = tools if tools else self.tools

        if use_slave:
            model = self._find_slave(self.model)

        # print(f"using model: {model}; use_slave: {use_slave}")
        # print(f"using system prompt: {system}")
        # print(f"using temperature: {temperature}")
        # print(f"using stream: {stream}")
        # print(f"using tools: {tools}")
        # print()
        
        # print(f"sending context: {context}")

        return self.client.chat.completions.create(
            model = model,
            messages = system + context,
            temperature = temperature,
            stream = stream,
            tools = tools
        )


    def _process_call(self, context, **kwargs):
        # print(f"Inside _process_call with: {context}, {kwargs}")
        
        user_query = self._get_latest_user_query(context = context)
        response = self._call_once(context = context, **kwargs)
        stream = kwargs.get("stream", False)

        text_response = ""
        if stream:
            if self.is_conditional_stream:
                return self._conditional_stream(response, user_query = user_query)

            else:
                # unconditional stream
                return self._unconditional_stream(response, user_query = user_query)
        
        else:
            # print("unstreamed response: ")
            # print(response)
            text_response = response.choices[0].message.content

            if response.choices[0].message.tool_calls:
                func_calls = response.choices[0].message.tool_calls # currently could not get the model to produce multiple functions in one go; if that happens you will have to loop through
                
                for func_call in func_calls:
                    args = func_call.function.arguments
                    func_name = func_call.function.name
                    text_response += getattr(self, func_name)(user_query, **args)

            return self._remove_stream_defs(text_response)


class BaseClaudeAgent(BaseAgent):

    def __init__(self, bundle):
        super().__init__(bundle = bundle)
      
        self.system = bundle["system_prompt"]
        self.client = self._init_client(base_url = self.base_url, key = os.getenv(bundle["key_env_variable"]))
        
        if not self.tools:
            self.tools = [] # Anthropic Client wants tools to be '[]' and not 'None'

        print(f"Initializing id with {self.id}\n")
        print(f"Initializing model with {self.model}\n")
        print(f"Initializing system_prompt with {self.system}\n")
        print(f"Initializing tools with {self.tools}\n")
        print(f"Initializing temperature with {self.temperature}\n")
        print(f"Initializing stream_def and default_behaviour with {self.stream_def}; {self.default_behaviour}\n")
        print("___________________________________________")


    def _init_client(self, base_url, key):
        if base_url:
            client = Anthropic(base_url = base_url, auth_token = key)
        else:
            client = Anthropic(auth_token = key)
        
        return client
    

    def _unconditional_stream(self, response, user_query):
        text_response, json_stream = "", ""
        total_stream = ""
        for chunk in response:
            if chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    text_response += chunk.delta.text
                    total_stream += chunk.delta.text
                    yield chunk.delta.text
                    
                elif chunk.delta.type == "input_json_delta":
                    json_stream += chunk.delta.partial_json
                    total_stream += chunk.delta.partial_json

            elif chunk.type == "content_block_start":
                # could be either 'text' or 'tool_use'; but here we only look for tool_use
                if chunk.content_block.type == "tool_use":
                    func_name = chunk.content_block.name
                    func_input = chunk.content_block.input

        if json_stream:
            # execute whatever function call
            json_stream = json.loads(json_stream)
            print(f"calling {func_name} with {json_stream} ...")
            # eval(func_name)(**json_response) 
    
            func_response = getattr(self, func_name)(user_query, **json_stream) # this function should also take care of processing calls again and should exist within the object
            for chunk in func_response:
                total_stream += chunk
                yield chunk
   

    def _conditional_stream(self, response, user_query):

        def process_stream(word_stream, current_behaviour):
            if current_behaviour == "blacklist":
                # pass
                return ""
            
            elif current_behaviour == "record":
                # this happens only if you've already hit a record function;
                streams["record_stream"] += word_stream[:x]
                return ""
            
            else:
                # whitelist                
                streams["allowed_stream"] += word_stream[:x]
                return word_stream[:x]


        def process_default_stream(word_stream, current_behaviour):
            if current_behaviour == "blacklist":
                return ""
            
            elif current_behaviour == "record":
                streams["record_stream"] += word_stream
                return ""
            
            else:
                # whitelist scenario
                # allowed_stream += word_stream
                streams["allowed_stream"] += word_stream
                return word_stream


        def process_remaining_stream(word_stream, current_behaviour):
            if current_behaviour == "blacklist":
                return ""
            
            elif current_behaviour == "record":
                # record_stream += word_stream[y:]
                streams["record_stream"] += word_stream[y:]
                return ""
            else:
                # allowed_stream += word_stream[y:]
                streams["allowed_stream"] += word_stream[y:]
                return word_stream[y:]

        
        streams = {"record_stream": "", "allowed_stream": ""}
        word_stream = ""
        current_behaviour = self.default_behaviour

        look_for_defs = list(self.stream_def.keys())
        found_closing_tag = False
        total_stream = []
        json_stream = ""

        for chunk in response:
            if chunk.type == "content_block_delta":
                if chunk.delta.type == "text_delta":
                    chunk = chunk.delta.text
                    total_stream.append(chunk) # present for debugging
                    word_stream += chunk
                    
                    if len(word_stream.split(" ")) >= self.lag_len:
                        # print(f"looking for {look_for_defs}")
                        analysis = analyse_stream(word_stream, self.stream_def, look_for_defs = look_for_defs, current_behaviour = current_behaviour, found_closing_tag = found_closing_tag)
                        if analysis:
                            # you must check for current behaviour right here and then update later
                        
                            x, y, closing_tag, found_closing_tag, look_for_defs = process_analysis(analysis, self.stream_def, return_behaviour=False) # ensure return_behaviour is False
                            
                            yield process_stream(word_stream, current_behaviour = current_behaviour)
                            
                            # check if there's another opening tag within the same stream. This happens more often than not  
                            _remaining_word_stream = word_stream[y:]
                            
                            # it is important to call 'current_behaviour' here in order to have these changes only AFTER encountering that tag. Till the opening tag has been encountered, behaviour must be the previous state/default
                            current_behaviour = analysis["behaviour"] if analysis["behaviour"] else self.default_behaviour
                            if not found_closing_tag:
                                current_behaviour = self.default_behaviour

                            if _remaining_word_stream:
                                remaining_analysis = analyse_stream(_remaining_word_stream, stream_def = self.stream_def, look_for_defs = look_for_defs, current_behaviour = current_behaviour, found_closing_tag = found_closing_tag)

                                if remaining_analysis:
                                    x, y, closing_tag, found_closing_tag, look_for_defs, current_behaviour = process_analysis(remaining_analysis, stream_def = self.stream_def, return_behaviour = True) # ensure return_behaviour is True

                                    # opening tag

                                    yield process_remaining_stream(word_stream=_remaining_word_stream, current_behaviour=current_behaviour)


                                else:
                                    yield process_remaining_stream(word_stream, current_behaviour)

                    
                        else:
                            # no tag in this stream, so continue with same behaviour the previous tag has
                            yield process_default_stream(word_stream, current_behaviour = current_behaviour)

                        word_stream = ""

                elif chunk.delta.type == "input_json_delta":
                    json_stream += chunk.delta.partial_json
                    total_stream += chunk.delta.partial_json


            elif chunk.type == "content_block_start":
                if chunk.content_block.type == "tool_use":
                    func_name = chunk.content_block.name
                    func_input = chunk.content_block.input

        if word_stream:
            # final check for closing tag or whatever
            remaining_analysis = analyse_stream(word_stream, stream_def = self.stream_def, look_for_defs = look_for_defs, current_behaviour = current_behaviour, found_closing_tag = found_closing_tag)
            if remaining_analysis:
                x, y, closing_tag, found_closing_tag, look_for_defs, current_behaviour = process_analysis(remaining_analysis, stream_def = self.stream_def, return_behaviour = True) # ensure return_behaviour is True

                yield process_stream(word_stream, current_behaviour=current_behaviour)

                _remaining_word_stream = word_stream[y:]
                        
                # it is important to call 'current_behaviour' here in order to have these changes only AFTER encountering that tag. Till the opening tag has been encountered, behaviour must be the previous state/default
                current_behaviour = remaining_analysis["behaviour"] if remaining_analysis["behaviour"] else self.default_behaviour
                if not found_closing_tag:
                    current_behaviour = self.default_behaviour

                if _remaining_word_stream:
                    remaining_analysis = analyse_stream(_remaining_word_stream, stream_def = self.stream_def, look_for_defs = look_for_defs, current_behaviour = current_behaviour, found_closing_tag = found_closing_tag)

                    if remaining_analysis:
                        x, y, closing_tag, found_closing_tag, look_for_defs, current_behaviour = process_analysis(remaining_analysis, stream_def = self.stream_def, return_behaviour = True) # ensure return_behaviour is True

                        # opening tag
                        yield process_remaining_stream(_remaining_word_stream, current_behaviour)
                        
                    
                    else:
                        yield process_remaining_stream(word_stream, current_behaviour)
                    
                    word_stream = ""
            # print(allowed_stream)

            else:
                yield process_default_stream(word_stream, current_behaviour = current_behaviour)

        # consider calling the function here itself. if there is a recorded stream
                

        # print(total_stream)
        print(f"FINALLL RECORD STREAM: {streams['record_stream']}")
       
        if json_stream:

            print(f"JSON STREAM: {json_stream} with func_name {func_name}")
            json_stream = json.loads(json_stream)
            print("ABOUT TO CALL")
            func_response = getattr(self, func_name)(user_query, **json_stream)

            # a hacky fix because func_response is a string (from the case law search function)

            if isinstance(func_response, str):
                yield func_response

            else:
                for chunk in func_response:
                    total_stream += chunk
                    yield chunk
    
        print(f"**************** {''.join(total_stream)}")
    


    def _call_once(self, context, system = "", model = "", tools = [], temperature = None, stream = False, use_slave = False):
        
        system = system if system else self.system
        model = model if model else self.model
        temperature = temperature if temperature else self.temperature
        tools = tools if tools else self.tools

        if use_slave:
            model = self._find_slave(self.model)

        # print(f"using model: {model}; use_slave: {use_slave}")
        # print(f"using system prompt: {system}")
        # print(f"using temperature: {temperature}")
        # print(f"using stream: {stream}")
        # print()
        
        # print(f"sending context: {context}")

        return self.client.messages.create(
            model = model,
            max_tokens = 4096 - 1,
            system = system,
            tools = tools,
            messages = context,
            temperature = temperature,
            stream = stream
            )
    

    def _process_call(self, context, **kwargs):
        # print(f"Inside _process_call with: {context}, {kwargs}")
        # call this function when you have a normal response to stream.
        
        # print(f"GETTING THIS AS CONTEXT: {context}")
        user_query = self._get_latest_user_query(context = context)


        response = self._call_once(context = context, **kwargs)
        stream = kwargs.get("stream", False)
        
        text_response = ""
        if stream:
            if self.is_conditional_stream:
                return self._conditional_stream(response, user_query = user_query)

            else:
                # unconditional stream
                return self._unconditional_stream(response, user_query = user_query)
        
        else:
            # you must adhere to stream_defs even in unconditional streams.
            # there seems to be another problem where if you just want to check for JSON stream and call the function later on. 
            text_response = response.content[0].text

            if self.is_conditional_stream:
                pass

            if len(response.content) > 1:
                func_call = response.content[1] # currently could not get the model to produce multiple functions in one go; if that happens you will have to loop through
                func_input, func_name = func_call.input, func_call.name
                # json_stream = json.loads(func_input)
                # func_response = getattr(self, func_name)(user_query, **json_stream)
                
                # json_stream = json.loads(func_input)
                func_response = getattr(self, func_name)(user_query, **func_input)
                text_response = text_response + "\n\n" + func_response

            return self._remove_stream_defs(text_response)



class SBERTEmbedder():
    def __init__(self, bundle) -> None:
        self.id = bundle["id"]
        self.model_path = bundle["model"]
        self.device = bundle["device"]
        self.truncate_dim = bundle["truncate_dim"]

        if not torch.cuda.is_available():
            print(f"{self.device} not found; continuing on CPU")
            self.device = "cpu"

        self.model = self.init_model()


    def init_model(self):
        if self.truncate_dim == -1:
            model = SentenceTransformer(self.model_path, device = self.device)
        else:
            # ensure model is trained with matryoshka loss before setting truncate_dim
            model = SentenceTransformer(self.model_path, device = self.device, truncate_dim = self.truncate_dim)
        return model


    def encode(self, queries):
        # queries can either be a single text of type str OR can be a list of strings
        return self.model.encode(queries, normalize_embeddings = True).tolist()

  
    def get_embedding(self, text):
        return self.encode(queries = text)

    # def __call__(self, queries):
    #     return self.encode(queries = queries)


class PineconeVDB():
    # def __init__(self, bundle, embedding_engine_bundle) -> None:
    def __init__(self, bundle, embedding_engine):

        self.id = bundle["id"]
        self.index_name = bundle["index_name"]

        # self.host_url = bundle["host_url"]
        self.client = Pinecone(api_key = os.getenv(bundle["key_env_variable"])) # maybe not pass this as an attribute ? directly init it ?
        self.metric = bundle.get("metric", "cosine")

        if "host_url" in bundle:
            self.host_url = bundle["host_url"]
            print(f"connecting with {self.host_url}")
            self.index = self.client.Index(host = self.host_url, metric = self.metric)

        else:
            self.index = self.client.Index(name = self.index_name, metric = self.metric)

        
        # self.index = self.client.Index(name = self.index_name, metric = self.metric)

        self.embedding_engine = embedding_engine
        self.index_type = bundle["index_type"]
        if self.index_type == "hybrid":
            self.sparse_encoder = BM25Encoder().load(bundle["sparse_model_path"])
            print(f"loaded sparse model from {bundle['sparse_model_path']}")


    def hybrid_score_norm(self, dense, sparse, alpha: float):
        """Hybrid score using a convex combination

        # alpha being 1 is purely semantic and alpha being 0 is purely sparse

        alpha * dense + (1 - alpha) * sparse

        Args:
            dense: Array of floats representing
            sparse: a dict of `indices` and `values`
            alpha: scale between 0 and 1
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        hs = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        return [v * alpha for v in dense], hs


    
    def retrieve(self, dense_embedding, sparse_embedding, namespace, alpha = 0.5,  top_k = 10):

        if self.index_type == "dense":
            # ignore sparse_embedding here
            results = self.index.query(
                namespace = namespace,
                vector = dense_embedding,
                top_k = top_k,
                include_metadata = True # it should always be set to True
            )
            print(f"returning dense results!")
        # print(f"done with retrieval {results}")

        elif self.index_type == "hybrid":
            
            dense_embedding, sparse_embedding = self.hybrid_score_norm(dense = dense_embedding, sparse = sparse_embedding, alpha = alpha)

            results = self.index.query(
                namespace = namespace,
                vector = dense_embedding,
                sparse_vector = sparse_embedding,
                top_k = top_k,
                include_metadata = True
            )
            print(f"returning hybrid results!")

        print(f"Raw responses: ")
        print(results, type(results))
        # write_json(results, "raw_response_hpc.json")
        return self.process_results(results)
        

    def process_results(self, results):
        processed_matches = []
        # print(f"RESULTS: {results}")
        for result in results["matches"]:
            metadata = result["metadata"]
            metadata["score"] = result["score"]
            processed_matches.append(metadata)

        # print(f"HERE: {processed_matches}")
        write_json(processed_matches, "processed_response_hpc.json")
        return processed_matches
    

    def rerank(self, results):
        results = sorted(results, key = lambda x: x["score"], reverse = True)
        unique_cases, visited = [], []

        for x in results:
            if x["case_id"] not in visited:
                visited.append(x["case_id"])
                unique_cases.append(x)

        return unique_cases


    def generate_embeddings(self, text):
        return self.embedding_engine.get_embedding(text = text) # by default always send normalized embeddings


    def retrieve_namespaces(self, query, namespaces, top_k = 10, alpha = 0.5, threshold = 0.15):
        # query_embedding = self.embedding_engine.get_embedding(text = query)
        print(f"using alpha of {alpha}")
        dense_embedding = self.generate_embeddings(text = query)
        sparse_embedding = self.sparse_encoder.encode_queries(query) if self.index_type == "hybrid" else {}
    
        with ThreadPoolExecutor() as executor:    
            search_results = list(executor.map(lambda namespace: self.retrieve(dense_embedding, sparse_embedding, namespace, alpha = alpha, top_k = top_k), namespaces))
        

        search_results = [result for ns_search_result in search_results for result in ns_search_result if result["score"] >= threshold]
        return self.rerank(search_results)

