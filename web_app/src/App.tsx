import '@mantine/core/styles.css'
import { MantineProvider } from '@mantine/core'
import { theme } from './theme'
import { DragDrop } from './components/DragDrop'
import { FileList } from './components/FileList'
import { useState } from 'react'
import { Button } from '@mantine/core'
import { IconSettingsCog } from '@tabler/icons-react'
import { QuestionTable } from './components/QuestionTable'
import { v4 as uuidv4 } from 'uuid'

import './App.css'
import {API_URL} from "./env.ts";

export interface Question {
  id: string
  question: string
}

export default function App() {
  const [files, setFiles] = useState<File[]>([])
  const [selection, setSelection] = useState<string[]>([])
  const [questions, setQuestions] = useState<Question[]>([])
  const [loading, setLoading] = useState<boolean>(false)

  const generateQuestions = async () => {
    setLoading(true)
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    try {
      const response = await fetch(`${API_URL}/generate_questions`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const data = await response.json()
        const questionsWithId = data.questions.map((question: string) => {
          return {
            id: uuidv4(),
            question,
          }
        })
        setQuestions(questionsWithId)
      }
    } catch (error) {
      console.error(error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <MantineProvider theme={theme}>
      <div className="body">
        <div className="upload-section">
          <DragDrop
            files={files}
            setFiles={setFiles}
          />

          <FileList
            files={files}
            setFiles={setFiles}
          />
        </div>

        <div className="generate-section">
          <Button
            loading={loading}
            onClick={generateQuestions}
            disabled={files.length === 0 || loading}
          >
            <IconSettingsCog /> Generate Questions
          </Button>
        </div>

        <div className="question-section">
          <QuestionTable
            selection={selection}
            setSelection={setSelection}
            questions={questions}
          />
        </div>
      </div>
    </MantineProvider>
  )
}